import time
import torch
import torch.nn as nn
import quantization
from torchvision import transforms as T
from tqdm import tqdm

def enable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):
            child.ptq = True
        else:
            enable_calibrate(child)
    return module

def disable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):  #AdaRoundQuantizer((observer): MinMaxObserver())  #AsymmetricQuantizer((observer): EMAMinMaxObserver())
            child.ptq = False     #别的都pass
        else:
            disable_calibrate(child)
    return module

def disable_soft_targets(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):
            child.soft_targets = False
        else:
            disable_soft_targets(child)
    return module

def enable_origin(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.convbn.QConv2dBn) or isinstance(child, quantization.linear.QLinear) or isinstance(child, quantization.conv.QConv2d):
            child.origin = True
        else:
            enable_origin(child)
    return module

def disable_origin(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.convbn.QConv2dBn) or isinstance(child, quantization.linear.QLinear) or isinstance(child, quantization.conv.QConv2d):
            child.origin = False
        else:
            disable_origin(child)
    return module

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Linear annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

def cal_recon_loss(orig, quan):
    return (torch.norm(quan - orig, p="fro", dim=1) ** 2).mean()
def cal_round_loss(round_vals, b):
    return (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

def is_leaf_module(module):

    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    if isinstance(module, quantization.convbn.QConv2dBn) or isinstance(module, quantization.linear.QLinear) or isinstance(module, quantization.conv.QConv2d):    
        return True
    else:
        return False
def run_hook_for_layers_with_given_input(model: torch.nn.Module, input_tensor,
                                         hook, module_type_for_attaching_hook=None, leaf_node_only=True):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_tensor: Input tensor to the model. If more than one model inputs, use a tuple
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :return: None
    """

    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = [module for module in model.modules() if not leaf_node_only or is_leaf_module(module)]
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]

    for module in modules:
        hooks.append(module.register_forward_hook(hook))

    # ------------------------------------------------
    # Run forward pass to execute the hook functions
    # ------------------------------------------------
    model.eval()
    with torch.no_grad():
        if isinstance(input_tensor, (list, tuple)):
            _ = model(*input_tensor)
        else:
            _ = model(input_tensor)
    model.train()
    # --------------------------
    # Remove all hooks we added
    # --------------------------
    for h in hooks:
        h.remove()

def get_ordered_list_of_modules(model: torch.nn.Module, dummy_input = torch.randn((1,3,224,224)).cuda()) :
    """
    Finds ordered modules in given model.
    :param model: PyTorch model.
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :return: List of module name, module in order.
    """
    def _hook_to_collect_name_of_module(module, input, output):
        """
        hook to find name of module
        """
        module_name = module_to_name_dict[module]
        list_modules.append([module_name, module])

    module_to_name_dict = {}

    # res = ['conv1','bn1'....]
    for name, module in model.named_modules():
        module_to_name_dict[module] = name

    list_modules = []
    run_hook_for_layers_with_given_input(model, dummy_input, hook=_hook_to_collect_name_of_module)

    return list_modules

class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """

def prepare_inout(model, module, model_input, collect_input=False, collect_output = False):
    def _hook_to_collect_inp_out_data(_, inp, out):
        """
        hook to collect input and output data
        """
        if collect_input:
            inp_data_list.append(inp[0])

        if collect_output:
            out_data_list.append(out)

        raise StopForwardException

    inp_data_list = []
    out_data_list = []

    handle = module.register_forward_hook(_hook_to_collect_inp_out_data)

    # get the model's device placement information
    device = torch.device("cuda")
    model.eval()
    # place the input to appropriate device
    if isinstance(model_input, torch.Tensor):
        model_input = model_input.to(device)
    else:
        for idx,img in enumerate(model_input):
            model_input[idx] = img.to(device)

    model = model.to(device)
    # Custom injected exception is raised when the activations data from desired module is collected.
    try:
        model(model_input)
    except StopForwardException:
        pass
    except AssertionError as e:
        print(str(e))
        pass
    model.train()
    handle.remove()
    torch.cuda.empty_cache()
    inp_data, out_data = None, None

    if inp_data_list and isinstance(inp_data_list[0], torch.Tensor):
        inp_data = inp_data_list[0].detach()

    if out_data_list and isinstance(out_data_list[0], torch.Tensor):
        out_data = out_data_list[0].detach()
    torch.cuda.empty_cache()
    return inp_data, out_data

def prepare_data_for_layer(trainloader, model, module, fast = False):
    origin = [] # output
    quant = [] # input
    enable_origin(model)
    for inputx in trainloader:            
        _ , origin_out = prepare_inout(model, module, inputx, collect_output = True)
        origin.append(origin_out.to('cpu'))
        torch.cuda.empty_cache()
    disable_origin(model)
    for inputx in trainloader:
        quant_in, _ = prepare_inout(model, module, inputx, collect_input = True)
        if fast:
            quant.append(quant_in)
        else:
            quant.append(quant_in.to('cpu'))
        torch.cuda.empty_cache()
    return quant, origin #input(hat(x)), output(wx)

def calibrate_adaround(model_name, model, adaround_iter, b_start, b_end, warmup, trainloader, device, logger=None, load_ckpt=False):
    logger.info("adaround_iter: {}".format(adaround_iter))   #这一部分就是在ti梯度下降求 公式25 21的 argmin
    # opt_params = []
    for name, child in model.named_modules():
        if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
            child.alpha.requires_grad = False
            # opt_params += [child.alpha]   #只优化这些Vi,j   #其实大部分都没有 就是conv和linear搞了    # print(opt_params)   # print('child.alpha: ', child.alpha)  #(64,3,7,7), (64,64,1,1)  最后有{list:54}
        if isinstance(child, quantization.convbn.QConv2dBn) or isinstance(child, quantization.linear.QLinear) or isinstance(child, quantization.conv.QConv2d):    
            child.weight.requires_grad=False
            if child.bias is not None:
                child.bias.requires_grad=False

    list_modules = get_ordered_list_of_modules(model)
    # logger.info(list_modules)
    # exit()
    from act import retinanet_mark_before_relu, resnet_mark_before_relu, deeplabv3_mark_before_relu
    if model_name == 'resnet':
        module_act_fundict = resnet_mark_before_relu(model)
    elif model_name == 'retinanet': #### retinanet会用很多次classification_head和regression_head，我们只用训练一次就行
        module_act_fundict = retinanet_mark_before_relu(model)
        newlist_modules = []
        already = []
        for mnames,thatmodule in list_modules:
            if mnames not in already:
                newlist_modules.append([mnames,thatmodule])
                already.append(mnames)
        list_modules = newlist_modules
    elif model_name == 'deeplabv3':
        module_act_fundict = deeplabv3_mark_before_relu(model)
        newlist_modules = []
        for mnames,thatmodule in list_modules:
            if 'aux_classifier' not in mnames:
                newlist_modules.append([mnames,thatmodule])
        list_modules = newlist_modules

    else:
        logger.error('unknown model')
        exit()
    disable_origin(model)
    if load_ckpt:
        thatall = torch.load(f"./ckpt/round_{model_name}.pth")
        that = thatall['net']
        dicts = model.state_dict()
        dicts.update(that)
        model.load_state_dict(dicts)
        allready_train = thatall['module_name']
        assert adaround_iter==thatall['adaround_iter']
        for idx, (name, module) in enumerate(list_modules):
            if is_leaf_module(module):
                module.weight_quantizer.alpha.requires_grad = False
            if name == allready_train:
                list_modules = list_modules[idx+1:]
                break
    
    for name, module in list_modules:
        module.weight_quantizer.alpha.requires_grad = True
        starttime = time.time()
        reconlist = []
        model = model.to(device)
        pre_inputdata, pre_outputdata = prepare_data_for_layer(trainloader, model, module)
        module.origin = False
        module.weight_quantizer.mode = True
        model = model.to('cpu')
        torch.cuda.empty_cache()
        module = module.to(device)
        optimizer = torch.optim.Adam([module.weight_quantizer.alpha])
        logger.info('start {}'.format(name))
        logger.info('alpha: {}'.format(module.weight_quantizer.alpha))
        temp_decay = LinearTempDecay(adaround_iter, rel_start_decay=warmup,start_b=b_start, end_b=b_end)
        # total_loss
        with tqdm(total = adaround_iter, leave=False, desc='adaround') as pbar:
            for j in range(adaround_iter):
                b = temp_decay(j)
                recon_loss_iter = []
                for idx in range(len(pre_inputdata)):                   
                    pre_input = pre_inputdata[idx].to(device)
                    orig_output = pre_outputdata[idx].to(device)
                    optimizer.zero_grad()
                    quan_output = module(pre_input)
                    act_func = module_act_fundict[name] if name in module_act_fundict.keys() else None
                    if act_func is not None:
                        quan_output = act_func(quan_output)
                        orig_output = act_func(orig_output)
                    recon_loss = cal_recon_loss(orig_output, quan_output)
                    that = recon_loss.detach().data.item() 
                    recon_loss_iter.append(that)
                    round_loss = cal_round_loss(module.weight_quantizer.get_soft_targets(), b)
                    total_loss = recon_loss + round_loss

                    # Back propagate and Update the parameter 'alpha'
                    total_loss.backward()
                    optimizer.step()
                recon_loss_iter = sum(recon_loss_iter)
                reconlist.append(recon_loss_iter)
                pbar.set_postfix({'recon_loss': recon_loss_iter, 'round_loss': round_loss.data.item()})
                pbar.update(1)
                
        logger.info('{} over! time:{} recon_loss_init:{}, recon_loss_last:{}'.format(name, time.time()-starttime, reconlist[0], reconlist[-1]))
        torch.save({'net':model.state_dict(), 'adaround_iter':adaround_iter, 'module_name':name}, f'./ckpt/round_{model_name}_wwq.pth')
        module.weight_quantizer.alpha.requires_grad = False
        module = module.to('cpu')
        logger.info('alpha: {}'.format(module.weight_quantizer.alpha))

    for name, child in model.named_modules():
        if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
            child.soft_targets = False   
  
def inplace_linear(linear,ptq):
    new_layer = quantization.QLinear(ptq,linear.in_features, linear.out_features,True if linear.bias is not None else False)
    new_layer.weight = linear.weight
    print(new_layer.weight.device)
    if linear.bias is not None:
        new_layer.bias = linear.bias
    return new_layer

def inplace_conv(conv, ptq):
    new_layer = quantization.QConv2d(ptq, conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                conv.padding, conv.dilation, conv.groups,True if conv.bias is not None else False)
    new_layer.weight = conv.weight
    if conv.bias is not None:
        new_layer.bias = conv.bias
    
    return new_layer

def inplace_conv_bn(conv, bn,   ptq):
    new_layer = quantization.QConv2dBn(ptq, bn,
                conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,conv.padding, conv.dilation, conv.groups,
                True if conv.bias is not None else False)
    new_layer.weight = conv.weight  #W参数 （63，3，7，7）
    if conv.bias is not None:
        new_layer.bias = conv.bias
    
    return new_layer
import torchvision
def inplace_quantize_layers(module, ptq, modulename = ''):
    last_conv_flag = 0   # adaround=True,level='L',ptq=True 其他都是false
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():  #第二遍的时候：child是Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  #Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
        if isinstance(child, (nn.modules.batchnorm._BatchNorm)) or isinstance(child, (torchvision.ops.FrozenBatchNorm2d)):  # conv+bn 组合的情况
            if last_conv is None:
                continue
            fused_qconv = inplace_conv_bn(last_conv, child, ptq)
            fused_qconv.weight_quantizer.layer_name = modulename + '.' + name
            module._modules[last_conv_name] = fused_qconv    # Conv2d 变成了 QConv2dBn
            module._modules[name] = nn.Identity()  #BatchNorm2d 变成了  Identity()  
            last_conv = None
            last_conv_flag = 0

        if last_conv_flag == 1:  # 纯conv 情况
            qconv = inplace_conv(last_conv, ptq)
            qconv.weight_quantizer.layer_name = modulename + '.' + last_conv_name
            module._modules[last_conv_name] = qconv
            last_conv = None
            last_conv_flag = 0 #retinanet 有很多这种情况

        if isinstance(child, nn.Conv2d):  #x
            
            
            last_conv = child   #Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            last_conv_name = name  #'conv1'
            last_conv_flag = 1  #标记上一层是conv层
            # print(list(child.named_children()))
            # exit(0)

      
            # if  child.named_children() is None:
            #     qconv = inplace_conv(last_conv, ptq)
            #     module._modules[last_conv_name] = qconv
            #     last_conv = None
            #     last_conv_flag = 0 #retinanet 有很多这种情况

        if isinstance(child, nn.Linear):  #x  线性层即全连接层
                qlinear = inplace_linear(child, ptq)
                qlinear.weight_quantizer.layer_name = modulename + '.' + name
                module._modules[name] = qlinear
        else:
            inplace_quantize_layers(child, ptq, modulename + '.' + name)  #会递归干净
            
    if last_conv_flag == 1:  # 纯conv 情况
        qconv = inplace_conv(last_conv, ptq)
        qconv.weight_quantizer.layer_name = last_conv_name
        module._modules[last_conv_name] = qconv
        last_conv = None
        last_conv_flag = 0 #retinanet 有很多这种情况
            
    return module
    
import logging
def setmylog(model_name):
    import os
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    if not os.path.exists('./ckpt'):
        os.mkdir('ckpt')
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    # 建立一个filehandler来把日志记录在文件里，级别为视args而定
    fh = logging.FileHandler(f'./logs/log_inference_{model_name}.log')
    fh.setLevel(logging.INFO)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为INFO以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    #将相应的handler添加在logger对象中
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info('')
    return logger


def adaround_infer(module, model_name):
    that = torch.load(f"./ckpt/round_{model_name}.pth")
    that = that['net']
    dicts = module.state_dict()
    dicts.update(that)
    module.load_state_dict(dicts)
    for name, child in module.named_modules():
        if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
            child.soft_targets = False   
import torch
import torch.nn as nn
import torchvision.ops.misc as misc # import FrozenBatchNorm2d


def absorb_bn(module, bn_module):  #module Conv2dWithId
    w = module.weight.data
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5) # bn eval 时的标准差分之一
    w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w)) # 64->(64,1,1,1)->(64,3,7,7)
    b.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine: # True, 可学习的bn层的weight和bias
        w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)
    # 置mean为0，one为1
    bn_module.register_buffer('running_mean', torch.zeros(module.out_channels).cuda())
    bn_module.register_buffer('running_var', torch.ones(module.out_channels).cuda())
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.affine = False

def absorb_bnf(module, bn_module):  #module Conv2dWithId
    w = module.weight.data
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5) # bn eval 时的标准差分之一
    w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w)) # 64->(64,1,1,1)->(64,3,7,7)
    b.add_(-bn_module.running_mean).mul_(invstd)
    w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
    b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    # 置mean为0，one为1
    bn_module.running_mean = torch.zeros(module.out_channels).cuda()
    bn_module.running_var = torch.ones(module.out_channels).cuda()
    bn_module.weight = None
    bn_module.bias = None

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) # nn.modules.batchnorm._BatchNorm
def is_bnf(m):
    return isinstance(m, misc.FrozenBatchNorm2d) # nn.modules.batchnorm._BatchNorm


def is_absorbing(m): # m是nn.Conv2d的话都是m.groups=1
    return (isinstance(m, nn.Conv2d) and m.groups == 1) or isinstance(m, nn.Linear)


def search_absorbe_bn(model):
    prev = None
    # import pdb
    # pdb.set_trace()
    for m in model.children():
        if is_absorbing(prev):
            if is_bn(m):   #线性/Conv+BN apply BNfolding on BN
                m.absorbed = True
                absorb_bn(prev, m)
            elif is_bnf(m):   #线性/Conv+BN apply BNfolding on BN
                m.absorbed = True
                absorb_bnf(prev, m)
        search_absorbe_bn(m)
        prev = m

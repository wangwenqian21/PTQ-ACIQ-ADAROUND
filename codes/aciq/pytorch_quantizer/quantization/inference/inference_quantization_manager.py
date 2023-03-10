import torch
import torch.nn as nn
from pytorch_quantizer.quantization import qtypes
from pytorch_quantizer.quantization.quantization_manager import QuantizationManagerBase
from itertools import count
import logging

logger = logging.getLogger(__file__)
VERBOSE = True

class ReLUWithId(nn.ReLU):
    _id = count(0)
    def __init__(self, inplace=False):
        super(ReLUWithId, self).__init__(inplace)

    def forward(self, input):
        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, '', "activation", half_range=True, verbose=QMI().verbose)  # gemmlowpMinMaxQuantize

        out = super(ReLUWithId, self).forward(input)  #就是做普通的Relu

        return out

class MaxPool2dWithId(nn.MaxPool2d):
    _id = count(0)
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2dWithId, self).__init__(kernel_size, stride, padding, dilation,return_indices, ceil_mode)
        self.id = next(self._id)

    def forward(self, input):
        out_id = 'maxpool%d_out' % self.id
        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, out_id, "activation_pooling", verbose=QMI().verbose)  # gemmlowpMinMaxQuantize

        out = super(MaxPool2dWithId, self).forward(input)

        if QMI().enabled:
            out = QMI().quantize_instant(out, out_id, "activation_pooling", verbose=QMI().verbose)  # gemmlowpMinMaxQuantize

        return out

class AvgPool2dWithId(nn.AvgPool2d):
    _id = count(0)
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2dWithId, self).__init__(kernel_size, stride, padding, ceil_mode,
                                              count_include_pad)
        self.id = next(self._id)

    def forward(self, input):
        out_id = 'avgpool%d_out' % self.id
        tag_act = 'activation_classifier' if input.shape[1] == 1000 else 'activation_pooling' # 'activation_classifier'和'activation_pooling'都是(minmax)

        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, out_id, tag_act, verbose=QMI().verbose)
        out = super(AvgPool2dWithId, self).forward(input)
        if QMI().enabled:
            out = QMI().quantize_instant(out, out_id, tag_act, verbose=QMI().verbose)

        return out

class AdaptiveAvgPool2dWithId(nn.AdaptiveAvgPool2d):
    _id = count(0)
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dWithId, self).__init__(output_size)
        self.id = next(self._id)

    def forward(self, input):
        out_id = 'avgpool%d_out' % self.id
        tag_act = 'activation_classifier' if input.shape[1] == 1000 else 'activation_pooling' # 'activation_classifier'和'activation_pooling'都是(minmax)

        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, out_id, tag_act, verbose=QMI().verbose)
        out = super(AdaptiveAvgPool2dWithId, self).forward(input)
        if QMI().enabled:
            out = QMI().quantize_instant(out, out_id, tag_act, verbose=QMI().verbose)

        return out

class Conv2dWithId(nn.Conv2d):
    _id = count(0)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWithId, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)
        self.id = next(self._id)   #0
        self.eps = torch.tensor([1e-8])
        # print('conv_%d' % self.id)

    def forward(self, input):
        # assert max([len(torch.unique(self.weight.data[idx,:,:,:]))for idx in range(self.weight.data.shape[0])])<=256
        activation_id = 'conv%d_activation' % self.id  #'conv0_activation'


        if input.shape[1]==3 and QMI().enabled:
            tag_act ='activation_first'
            input = QMI().quantize_instant(input, activation_id, tag_act, verbose=QMI().verbose)  #好像非常符合第二篇综述的过程，卷完了再对activation做quantize

        out = super(Conv2dWithId, self).forward(input)  #(64,64,112,112)

        tag_act = 'activation_classifier' if out.shape[1] == 1000 or out.shape[1] == 21 else 'activation' # 如果conv是最后一层，就用'activation_classifier'(minmax)，否则用'activation'(clipping，perchannel)
        if tag_act == 'activation_classifier':
            return out
        if QMI().enabled:
            out = QMI().quantize_instant(out, activation_id, tag_act, half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)  #好像非常符合第二篇综述的过程，卷完了再对activation做quantize

        return out


class LinearWithId(nn.Linear):
    _id = count(0)

    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithId, self).__init__(in_features, out_features, bias)
        self.id = next(self._id)

    def forward(self, input):
        # if QMI().
        assert max([len(torch.unique(self.weight.data[idx,:]))for idx in range(self.weight.data.shape[0])])<=256
        activation_id = 'linear%d_activation' % self.id
        # tag_act = 'activation_linear'
        #### 最后一层后的activation不量化
        tag_act = 'activation_classifier' if self.weight.shape[0] == 1000 else 'activation_linear' # 如果linear是最后一层，就用'activation_classifier'(minmax)，否则用'activation_linear'(clipping, 非perchannel)
        
        half_range = hasattr(self, 'before_relu') if self.weight.shape[0] != 1000 else False  #False

        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, activation_id, tag_act, verbose=QMI().verbose)
        out = super(LinearWithId, self).forward(input)
        if tag_act == 'activation_classifier':
            return out
        if QMI().enabled:
            out = QMI().quantize_instant(out, activation_id, tag_act, half_range = half_range, verbose=QMI().verbose)

        return out


# TODO: batch norm folding
class BatchNorm2dWithId(nn.BatchNorm2d):
    _id = count(0)
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dWithId, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.id = next(self._id)
        # print('bn_%d' % self.id)

    def forward(self, input):
        activation_id = 'bn%d_activation' % self.id
        if QMI().bn_folding and hasattr(self, 'absorbed'):
            return input   # 直接return input 因为已经bn folding了
        else:
            import pdb
            pdb.set_trace()
        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, activation_id, "activation", half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)

        out = super(BatchNorm2dWithId, self).forward(input)

        if QMI().enabled:
            out = QMI().quantize_instant(out, activation_id, "activation", half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)


        return out

import torchvision.ops.misc as misc

class FrozenBatchNorm2dWithId(misc.FrozenBatchNorm2d):
    _id = count(0)
    def __init__(self, num_features, eps = 1e-5):
        super(FrozenBatchNorm2dWithId, self).__init__(num_features, eps)
        self.id = next(self._id)
        # print('bn_%d' % self.id)

    def forward(self, input):
        activation_id = 'bn%d_activation' % self.id
        if QMI().bn_folding and hasattr(self, 'absorbed'):
            return input   # 直接return input 因为已经bn folding了
        else:
            import pdb
            pdb.set_trace()

        # if QMI().enabled:
        #     input = QMI().quantize_instant(input, activation_id, "activation", half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)

        out = super(FrozenBatchNorm2dWithId, self).forward(input)
        if QMI().enabled:
            out = QMI().quantize_instant(out, activation_id, "activation", half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)

        return out


class QuantizationManagerInference(QuantizationManagerBase):
    def __init__(self, args, qparams):
        super(QuantizationManagerInference, self).__init__()
        self.args = args
        self.verbose = False
        self.quantize = args.qtype is not None
        self.disable_quantization = args.q_off
        self.op_manager = TruncationOpManagerInference(args, qparams)
        self.enabled = False
        self.bn_folding = False
        self.bcorr_act = args.bias_corr_act
        self.bcorr_weight = args.bias_corr_weight
        self.vcorr_weight = args.var_corr_weight
        self.bmu = []

    def __exit__(self, *args):
        self.op_manager.__exit__(args)
        super(QuantizationManagerInference, self).__exit__(args)

    def quantize_instant(self, tensor, id, tag="", half_range=False, override_att=None, verbose=False):
        return self.op_manager.quantize_instant(tensor, id, tag, half_range, override_att, verbose)  #self.op_manager：TruncationOpManagerInference

    def reset_counters(self):
        ReLUWithId._id = count(0)
        pass

    def quantize_model(self, model):   #量化参数
        # import pdb'
        # pdb.set_trace()'
        for n, m in model.named_modules(): # ['', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.relu', 'layer1.0.downsample', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.relu', 'layer1.2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.relu', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.relu', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.relu', 'layer2.2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.relu', 'layer2.3', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.relu', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.relu', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.relu', 'layer3.2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.relu', 'layer3.3', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.relu', 'layer3.4', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.relu', 'layer3.5', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.relu', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.relu', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.relu', 'layer4.2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.relu', 'avgpool', 'fc']
            weight_q = None
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):   #只有Conv和Linear 做量化，yes
                if m.weight.shape[1]==3 or m.weight.shape[0]==1000 or m.weight.shape[0]==21:
                    tag = 'weight_classifier'
                else:
                    tag = 'weight'
                weight_q = QMI().quantize_instant(m.weight, n + '.weight', tag, verbose=True)

            if weight_q is not None:
                if self.vcorr_weight or self.bcorr_weight:
                    bias_q = weight_q.view(weight_q.shape[0], -1).mean(-1)
                    bias_q = bias_q.view(bias_q.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_q.view(bias_q.numel(), 1)
                    bias_orig = m.weight.view(m.weight.shape[0], -1).mean(-1)
                    bias_orig = bias_orig.view(bias_orig.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_orig.view(bias_orig.numel(), 1)

                if self.vcorr_weight:
                    eps = torch.tensor([1e-8]).to(weight_q.device)
                    var_corr = m.weight.view(m.weight.shape[0], -1).std(dim=-1) / \
                            (weight_q.view(weight_q.shape[0], -1).std(dim=-1) + eps)
                    var_corr = (var_corr.view(var_corr.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else var_corr.view(var_corr.numel(), 1))

                    # Correct variance
                    weight_q = (weight_q - bias_q) * var_corr + bias_q

                if self.bcorr_weight:
                    # Correct mean
                    weight_q = weight_q - bias_q + bias_orig
                m.weight.data = weight_q

    def plotbmu(self):
        import json
        import matplotlib.pyplot as plt
        plt.hist(self.op_manager.bmu)
        plt.savefig('bum{}'.format(self.args.arch))
        import pdb
        pdb.set_trace()
        with open('bum{}.json'.format(self.args.arch),'w')as fp:
            json.dump(self.op_manager.bmu, fp=fp)

# Alias
QMI = QuantizationManagerInference

class TruncationOpManagerInference:
    def __load_quantizer__(self, qtype, qparams):
        qtype_name = qtype.rstrip('1234567890')
        quant_params = qparams[qtype_name] if qtype_name in qparams else {}
        quantizer = qtypes.__dict__[qtype_name + "_quantizer"](qtype, quant_params) # int_quantizer
        return quantizer, quant_params

    def __fill_quantizers__(self, qtype, qparams, arch=None, qweight='int8'):
        weights_quantizer, _ = self.__load_quantizer__(qweight, qparams) # gemmlowpQuantizeWeightsPerChannel
        weights_quantizer.pcq_a = False
        weights_quantizer.clipping = 'no'
        self.quantizers['weight'] = weights_quantizer

        weights_quantizer, _ = self.__load_quantizer__('int8', qparams) # gemmlowpQuantizeWeightsPerChannel
        weights_quantizer.pcq_a = False
        weights_quantizer.clipping = 'no'
        self.quantizers['weight_classifier'] = weights_quantizer

        activation_quantizer, _ = self.__load_quantizer__(qtype, qparams) # gemmlowpClippingQuantize
        activation_quantizer.pcq_w = False
        self.quantizers['activation'] = activation_quantizer

        activation_linear_quantizer, _ = self.__load_quantizer__(qtype, qparams) # gemmlowpClippingQuantize
        activation_linear_quantizer.pcq_w = False
        activation_linear_quantizer.pcq_a = False # 非perchannel
        self.quantizers['activation_linear'] = activation_linear_quantizer # linear后的用aciq也不用perchannel

        classifier_quantizer, _ = self.__load_quantizer__('int8', qparams) # gemmlowpMinMaxQuantize
        classifier_quantizer.clipping = 'no'
        classifier_quantizer.pcq_w = False
        classifier_quantizer.pcq_a = False
        self.quantizers['activation_classifier'] = classifier_quantizer

        first_quantizer, _ = self.__load_quantizer__('int8', qparams) # gemmlowpMinMaxQuantize
        first_quantizer.clipping = 'no'
        first_quantizer.pcq_w = False
        self.quantizers['activation_first'] = first_quantizer

        # Pooling is currently not working well with clipping. Leave it in 8 bit.
        pooling_quantizer, _ = self.__load_quantizer__('int8', qparams) # gemmlowpMinMaxQuantize
        pooling_quantizer.pcq_w = False
        pooling_quantizer.pcq_a = False
        pooling_quantizer.clipping = 'no'
        self.quantizers['activation_pooling'] = pooling_quantizer

    def __init__(self, args, qparams):
        self.verbose = False
        self.activation_quantizer = None
        self.origin_linear = nn.Linear
        self.origin_conv2d = nn.Conv2d
        self.origin_batch_norm = nn.BatchNorm2d
        self.orig_maxpool = nn.MaxPool2d
        self.orig_avgpool = nn.AvgPool2d
        self.orig_adtavgpool = nn.AdaptiveAvgPool2d
        self.orig_relu = nn.ReLU
        self.origin_batch_norm_frozen = misc.FrozenBatchNorm2d
        self.ignore_ids = []
        self.bmu = []
        self.quantizers = {}
        self.quantize = True
        self.__fill_quantizers__(args.qtype, qparams, args.arch, args.qweight)
        self.quantizer_default, _ = self.__load_quantizer__('int8', qparams)

    def __exit__(self, *args):
        pass

    def get_quantizer(self, tag, tensor=None):
        if tag in self.quantizers:
            return self.quantizers[tag]
        else:
            return self.quantizer_default

    def enable(self):
        nn.Linear = LinearWithId
        nn.Conv2d = Conv2dWithId
        nn.BatchNorm2d = BatchNorm2dWithId
        nn.MaxPool2d = MaxPool2dWithId
        nn.AvgPool2d = AvgPool2dWithId
        nn.AdaptiveAvgPool2d = AdaptiveAvgPool2dWithId
        nn.ReLU = ReLUWithId
        misc.FrozenBatchNorm2d = FrozenBatchNorm2dWithId
        

    def disable(self):
        nn.Linear = self.origin_linear
        nn.Conv2d = self.origin_conv2d
        nn.BatchNorm2d = self.origin_batch_norm
        nn.MaxPool2d = self.orig_maxpool
        nn.AvgPool2d = self.orig_avgpool
        nn.AdaptiveAvgPool2d = self.orig_adtavgpool
        nn.ReLU = self.orig_relu
        misc.FrozenBatchNorm2d = self.origin_batch_norm_frozen


    def quantize_instant(self, tensor, id, tag="", half_range=False, override_att=None, verbose=False):
        # ignore quantization of first and last layer
        qtag = tag   #conv："weight"
        q = self.get_quantizer(qtag)  #conv和linear的都是这个 IntQuantizer - [bits: 8, clipping: no, bit_alloc_act: True, bit_alloc_weight: True, bit_alloc_round: True, pcq_w: True, pcq_a: False, bcorr_act: False, bcorr_weight: True, vcorr_weight: False, kind: max]
        q.half_range = half_range   #activation 也是IntQuantizer
        # import pdb
        # pdb.set_trace()
        res = q(tensor, id, tag, override_att)

        # self.bmu.extend(q.bmu)
        # q.clearbmu()
        return res  #weight: gemmlowpQuantizeWeightsPerChannel   #activation: gemmlowpClippingQuantize

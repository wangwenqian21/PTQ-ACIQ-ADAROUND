from . import quantizer
from . import observer

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class QConv2dBn(nn.Conv2d):
    def __init__(self, ptq, bn, in_channels, out_channels, kernel_size, stride=1, padding=0,  #bn:BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                dilation=1, groups=1, bias=True,padding_mode='zeros', bit=8,sign=True, **kwargs):  #传参只传到bias

        super(QConv2dBn, self).__init__(in_channels, out_channels, kernel_size, stride, padding,dilation, groups, bias, padding_mode)
        
        self.register_buffer("running_mean", copy.deepcopy(bn.running_mean))
        self.register_buffer("running_var", copy.deepcopy(bn.running_var))
        self.gamma = copy.deepcopy(bn.weight)
        self.beta = copy.deepcopy(bn.bias)
        self.eps = bn.eps
        self.ptq = ptq
        self.origin= False
        self.weight_quantizer = quantizer.AdaRoundQuantizer(bit = bit, observer = observer.MinMaxObserver(), ptq = ptq)
        
        self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, observer = observer.EMAMinMaxObserver(),ptq = ptq)
        
    def forward(self, input):
        if not self.origin:
            if self.bias is not None:  # BN folding
                bias_fused = (self.beta + (self.bias - self.running_mean)* (self.gamma / torch.sqrt(self.running_var + self.eps))).reshape(-1)   #（8，）
            else:
                bias_fused = (self.beta- self.running_mean* (self.gamma / torch.sqrt(self.running_var + self.eps))).reshape(-1)  #  Parameter(64,) #（64，） 对应white论文里公式10

            weight_fused = self.weight * (self.gamma / torch.sqrt(self.running_var + self.eps)).reshape(-1, 1, 1, 1)   #self.weight(64,3,7,7)  weight_fused(64,3,7,7) 对应white论文里公式10

            input = self.input_quantizer(input)  #得到量化后的x #input: (128,3,224,224)  原本 (128,1,28,28)     #self.input_quantizer =  AsymmetricQuantizer((observer): EMAMinMaxObserver())   #第二轮：(128,8,14,14)   identity,relu,max_pool debug的时候都没有停留

            weight_quant = self.weight_quantizer(weight_fused)  # fake_quant_tensor   #整的是weight #self.weight_quantizer =  AdaRoundQuantizer((observer): MinMaxObserver())   observer 不一样  Quantizer也不一样
            #weight_fused: tensor([[[[ 7.0902e-01,  9.6848e-01,  7.0332e-01],[-7.7668e-01, -2.8736e-01,  9.4384e-01],[-7.0567e-01, -1.2511e+00, -1.1366e+00]]],） weight_quant:tensor([[[[ 0.7105,  0.9634,  0.6985], [-0.7708, -0.2890,  0.9394],  [-0.7105, -1.2525, -1.1320]]]

            output = F.conv2d( input, weight_quant, bias_fused, self.stride, self.padding,   # bias,If True, adds a learnable bias to the output. Default: True
                self.dilation, self.groups)   #"改了"input和weight 还是输入F.conv2d
            
            #self.storeqwx=output
            
            return output
        
        else:
            if self.bias is not None:  # BN folding
                bias_fused = (self.beta + (self.bias - self.running_mean)* (self.gamma / torch.sqrt(self.running_var + self.eps))).reshape(-1)   #（8，）
            else:
                bias_fused = (self.beta- self.running_mean* (self.gamma / torch.sqrt(self.running_var + self.eps))).reshape(-1)  #  Parameter(64,) #（64，） 对应white论文里公式10
                
            weight_fused = self.weight * (self.gamma / torch.sqrt(self.running_var + self.eps)).reshape(-1, 1, 1, 1)   #self.weight(64,3,7,7)  weight_fused(64,3,7,7) 对应white论文里公式10
            output = F.conv2d( input, weight_fused, bias_fused, self.stride, self.padding,   # bias,If True, adds a learnable bias to the output. Default: True
                        self.dilation, self.groups)   #"改了"input和weight 还是输入F.conv2d
            
            #self.storewx=output
            
            return output            
    
    # def get_storeqwx(self):
    #     return self.storeqwx
        
    # def get_storewx(self):
    #     return self.storewx
        
        
        
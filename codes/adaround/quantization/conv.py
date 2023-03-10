from . import observer
from . import quantizer

import torch.nn as nn
import torch.nn.functional as F

class QConv2d(nn.Conv2d):
    def __init__(self, ptq,in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1,
                 groups=1, bias=True,padding_mode='zeros', bit=8, **kwargs):
        super(QConv2d, self).__init__( in_channels, out_channels, kernel_size, stride, padding,dilation, groups, bias, padding_mode)
        self.ptq = ptq
        self.origin= False
        self.weight_quantizer = quantizer.AdaRoundQuantizer(bit = bit, observer = observer.MinMaxObserver(),ptq = ptq)
        self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, observer = observer.EMAMinMaxObserver(),ptq = ptq)

    def forward(self, input):
        if not self.origin:
            input = self.input_quantizer(input)
            weight_quant = self.weight_quantizer(self.weight)

            output = F.conv2d(input, weight_quant, self.bias, self.stride, self.padding,self.dilation, self.groups)
            #self.storeqwx=output
            return output
        
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,self.dilation, self.groups)
            #self.storewx=output
            return output 
    
    # def get_storeqwx(self):
    #     return self.storeqwx
    
    # def get_storewx(self):
    #     return self.storewx
    



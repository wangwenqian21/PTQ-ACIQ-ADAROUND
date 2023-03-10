from . import quantizer
from . import observer

import torch.nn as nn
import torch.nn.functional as F

class QLinear(nn.Linear):
    def __init__(self, ptq,in_features, out_features, bias=True, bit=8, **kwargs):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.ptq = ptq
        self.origin= False
        self.weight_quantizer = quantizer.AdaRoundQuantizer(bit = bit, observer = observer.MinMaxObserver(),ptq = ptq)
        self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit,observer = observer.EMAMinMaxObserver(),ptq = ptq)

    def forward(self, input):
        if not self.origin:
            input = self.input_quantizer(input)
            weight_quant = self.weight_quantizer(self.weight)
            output = F.linear(input, weight_quant, self.bias)  #(128,10)
            #self.storeqwx=output
            return output

        else:
            output = F.linear(input, self.weight, self.bias)  #(128,10)
            #self.storewx=output
            return output        
            
    # def get_storeqwx(self):
    #     return self.storeqwx
    
    # def get_storewx(self):
    #     return self.storewx
import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, bit, observer, ptq):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
    
    def update_qparams(self, tensor):
        raise NotImplementedError
    
    def forward(self, tensor):
        if self.ptq:
            self.observer(tensor)   #EMAMinMaxObserver   记录了下input的最大最小值
            self.update_qparams(tensor)   #调整 scale 和 zero point参数
        torch.cuda.empty_cache()
        quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point   #实际是：(torch.round(tensor / self.scale)+ self.zero_point  #xint  我证了 scale和zero point的选择可以正好让xint 落在[qmin,qmax]区间，不用clamp  #(128,1,28,28)
        fake_quant_tensor = self.scale * (quant_tensor - self.zero_point)    #(128,1,28,28)  #dequantization
        # tensor 本来是 ([-0.5082, -0.3883, -0.4226...]) quant_tensor = 都变成整数([ -41.,  -35.,  -37.,  ...,.， fake_quant_tensor [-0.5038, -0.3918, -0.4291, 又变成小数，而且确实和tensor有细微差别
        torch.cuda.empty_cache()
        return fake_quant_tensor
  
class AsymmetricQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq

        self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),)
    
    def update_qparams(self, inputs):
        #对应 INTEGER 论文 公式(1)(2)
        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)  
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + (self.quant_min - self.observer.min_val / scale)  #round 四舍五入 torch.round() 不能求导 所以用detach() 写法 #detach(): Returns a new Tensor, detached from the current graph. The result will never require gradient.  

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


class AdaRoundQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq ):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer  # MinMaxObserver()
        self.ptq = ptq
        self.alpha = None
        self.ada_init = None
        self.soft_targets = True
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("quant_min",torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),)
        self.register_buffer("quant_max",torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),)
    
    def update_qparams(self, inputs):
        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                        (self.quant_min - self.observer.min_val / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    def forward(self, tensor):
        if self.ptq:
            self.observer(tensor)  
            self.update_qparams(tensor)

        if not self.ada_init:
            self.init_alpha(tensor.clone())   #初始化 Vi,j
            self.ada_init = True
        torch.cuda.empty_cache()
        #第二个batch就走这了
        quant_tensor = self.quant(tensor)   #tensor([[[[  56.,   77.,   55.], [ -67.,  -27.,   75.], [ -62., -107.,  -97.]]],
        fake_quant_tensor = self.dequantize(quant_tensor)  #tensor([[[[ 0.7105,  0.9634,  0.6985], [-0.7708, -0.2890,  0.9394],  [-0.7105, -1.2525, -1.1320]]],
        torch.cuda.empty_cache()
        return fake_quant_tensor
    

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point

        x_floor = torch.floor(inputs / scale)

        if self.soft_targets:
            x_ada = x_floor + self.get_soft_targets()  # 完整的是公式22的一部分clamp在底下 #self.get_soft_targets()是标准的公式23 h(V i,j)
        else:
            #print('test test test')
            x_ada = x_floor + (self.alpha >= 0).float()    # (self.alpha >= 0).float() torch.float32  全是0，1
        #这块决定到底是 向下取整还是向上取整

        outputs = x_ada + zero_point

        # outputs = outputs.clamp(self.quant_min,self.quant_max)
        outputs = ( (outputs.round() - outputs ).detach() + outputs  ).clamp(self.quant_min,self.quant_max)
        torch.cuda.empty_cache()
        return outputs  #得到Wint

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point

        outputs = (inputs - zero_point) * scale
        return outputs
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)  #h(V i,j)  self.alpha 就是Vi,j

    def init_alpha(self, x: torch.Tensor):
        scale = self.scale
        x_floor = torch.floor(x / scale)
    
        #print('Init alpha to be FP32')
        rest = (x / scale) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # 初始化的时候 使得h(Vi,j)= w/s - floor(w/s)=rest 解这个方程得到 sigmoid（Vi,j)的表达式   公式23 省去clip   # => sigmoid(alpha) = rest
        self.alpha = nn.Parameter(alpha)
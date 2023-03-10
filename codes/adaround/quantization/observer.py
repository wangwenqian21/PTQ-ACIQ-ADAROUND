import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):  #input:128,3,224,224  # layer级(activation/weight)

        min_val = torch.min(input) # 找到了最小值
        max_val = torch.max(input) #找到最大值

        self.update_range(min_val, max_val)   #max_val=1,min_val=0   更新了 self.min_val和self.max_val
        return input


class MinMaxObserver(ObserverBase):
    def __init__(self):
        super(MinMaxObserver, self).__init__()
        self.num_flag = 0
 
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)  #一直保持一个最大最小值，不断更新
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class EMAMinMaxObserver(ObserverBase):
    def __init__(self, momentum=0.1):
        super(EMAMinMaxObserver, self).__init__()
        self.momentum = momentum
        self.num_flag = 0

        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))


    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:  
            self.num_flag += 1
            min_val = min_val_cur  #tensor(-2.1179, device='cuda:0')  因为num_flag是0 所以没实现底下那种
            max_val = max_val_cur  #tensor(2.6400, device='cuda:0')
        else:  #第二个batch 就会跑到这里
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)



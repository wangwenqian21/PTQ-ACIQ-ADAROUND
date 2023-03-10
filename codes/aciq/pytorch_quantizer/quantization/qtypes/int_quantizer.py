import torch
from torch.autograd import Function
import numpy as np
# Alpha coeficients for for laplace clipping
# [2.83068299 3.89722946 5.02864014 6.20476633 7.41312622 8.64561995 9.89675982]
def to_cuda(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        return torch.tensor(t, dtype=torch.float32).to(device)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor

class IntQuantizer(Function):
    def __init__(self, size, params):
        self.num_bits = size
        # TODO: expose as cmd line parameters
        self.enforce_true_zero = params['true_zero'] # True
        self.clipping = params['clipping'] if 'clipping' in params else 'no'
        self.pcq_w = params['pcq_weights']
        self.pcq_a = params['pcq_act']
        self.bcorr_act = params['bcorr_act']
        self.bcorr_weight = params['bcorr_weight']
        self.vcorr_weight = params['vcorr_weight']
        self.logger = params['logger']
        self.bmu = []

        # self.alpha_gaus = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        # self.alpha_gaus_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

        self.alpha_laplace = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        self.alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16269} # half range, before relu做的
        
        # 8 bit: alpha_laplace=9.89; alpha_laplace_positive=11.16
        # self.gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
        self.half_range = False
    # def clearbmu(self):
    #     self.bmu.clear()

    def __call__(self, tensor, id, tag="", override_att=None):
        if override_att is not None:
            orig_att = getattr(self, override_att[0])
            setattr(self, override_att[0], override_att[1])
        elif self.clipping != 'no': # activation, activation_linear
            # print("clipping %s: %d" % (tag, self.num_bits))
            res = self.gemmlowpClippingQuantize(tensor, id, tag, clip_type=self.clipping)
        elif self.pcq_w:   # weight
            # print("pcq_w %s: %d" % (tag, self.num_bits))
            res = self.gemmlowpQuantizeWeightsPerChannel(tensor, id)
        elif self.pcq_a: # activation before first layer
            res = self.gemmlowpQuantizeActivationPerChannel(tensor, id)
        else: # activation_classifier, activation_pooling
            # print("no clipping %s: %d" % (tag, self.num_bits))
            res = self.gemmlowpMinMaxQuantize(tensor, tag)

        if override_att is not None:
            setattr(self, override_att[0], orig_att)
        return res

    def __repr__(self):
        return 'IntQuantizer - [bits: {}, clipping: {}, pcq_w: {}, pcq_a: {}, bcorr_act: {}, bcorr_weight: {}, vcorr_weight: {}]'\
            .format(self.num_bits, self.clipping, self.pcq_w, self.pcq_a, self.bcorr_act, self.bcorr_weight, self.vcorr_weight)

    def get_alpha_laplace(self, tensor, mean, per_channel=False):
        if per_channel:
            b = self.__act_stats_perchannel__(tensor, ['b'], avg_over_batch=False)['b']
        else:
            b = self.__act_stats__(tensor, ['b'], avg_over_batch=False)['b']
        aciq_factor = (self.alpha_laplace_positive[self.num_bits] if self.half_range else self.alpha_laplace[self.num_bits])

        return to_cuda(b, tensor.device) * aciq_factor, None
    def alpha2DeltaOffset(self, alpha, max_value, min_value, mean, mask, clip2max=False):
        alpha = to_numpy(alpha)
        max_value = to_numpy(max_value)
        min_value = to_numpy(min_value)
        mean = to_numpy(mean)
        if self.half_range:
            # import pdb
            # pdb.set_trace()
            delta = np.maximum(np.array(mean), 0) + alpha
            offset = 0
            if clip2max:
                delta = np.minimum(delta, max_value)
        else:
            delta = 2 * alpha # 2a/2^M->s
            ##### 感觉这样才是对的
            offset = np.maximum(min_value, mean - alpha) # min处的截断点
            if clip2max: # offset+delta 不超过max
                delta = np.minimum(delta, max_value - offset)

        return delta, offset

    ######## 我们试图用附加项对fused relu进行矫正，但结果很差，可能存在一定问题，还未找到修正的方式 ########
    ## def tiao(self,d):
    ##     k= 0.08
    ##     return k*(1.046*d-0.0262*abs(d)**2.42-0.02)
    # def tiao(self,d, M = 8):
    #     if M == 8:
    #         k = -0.92
    #         return k*(0.996*d+0.011*abs(d)**0.65+0.001*d**2-0.0045)
    #     if M ==4:
    #         k = -0.89
    #         return k*(0.963*d+0.019*abs(d)**1.3+0.001*d**2-0.008)
    # def get_alpha_laplace(self, tensor, mean, per_channel=False):
    #     if per_channel:
    #         b = self.__act_stats_perchannel__(tensor, ['b'], avg_over_batch=False)['b']
    #         mub = mean/b
    #         # import pdb
    #         # pdb.set_trace()
    #         # mask1 = abs(mub)<3 
    #         if self.half_range:
    #             mask = abs(mub)<=3 
    #             tiaomub = self.tiao(mub,self.num_bits)
    #             res = to_cuda(b, tensor.device) * ((self.alpha_laplace_positive[self.num_bits]-tiaomub)*mask.float() + self.alpha_laplace_positive[self.num_bits]*(1-mask.float()))
    #             mask = mask * (self.alpha_laplace_positive[self.num_bits]-tiaomub)
    #         else:
    #             res = to_cuda(b, tensor.device) * self.alpha_laplace[self.num_bits]
    #             mask = None
    #     else:
    #         b = self.__act_stats__(tensor, ['b'], avg_over_batch=False)['b']
    #         # self.bmu.append((b/self.__act_stats__(tensor, ['mean'], avg_over_batch=False)['mean']).detach().cpu())
    #         mub = mean/b
    #         if self.half_range:
    #             if abs(mub)<=3:
    #                 tiaomub = self.tiao(mub,self.num_bits)
    #                 res = to_cuda(b, tensor.device) * (self.alpha_laplace_positive[self.num_bits]-tiaomub)
    #                 mask = (self.alpha_laplace_positive[self.num_bits]-tiaomub)
    #             else:
    #                 res = to_cuda(b, tensor.device) * self.alpha_laplace_positive[self.num_bits]
    #                 mask = 0
    #         else:
    #             res = to_cuda(b, tensor.device) * self.alpha_laplace[self.num_bits]
    #             mask = None
    #     return res, mask

    # def alpha2DeltaOffset(self, alpha, max_value, min_value, mean, mask, clip2max=False):
    #     alpha = to_numpy(alpha)
    #     max_value = to_numpy(max_value)
    #     min_value = to_numpy(min_value)
    #     mean = to_numpy(mean)
    #     if mask is not None:
    #         mask = to_numpy(mask)
    #     if self.half_range:
    #         # import pdb
    #         # pdb.set_trace()
    #         delta = mask + (np.maximum(0, mean)+alpha)*(mask==0)
    #         # offset = 0*mask + np.maximum(min_value, mean - alpha)*(mask==0)
    #         offset = 0
    #         if clip2max:
    #             delta = np.minimum(delta, max_value - offset)
    #     else:
    #         delta = 2 * alpha # 2a/2^M
    #         ##### 感觉这样才是对的
    #         offset = np.maximum(min_value, mean - alpha) # min处的截断点
    #         if clip2max: # offset+delta 不超过max
    #             delta = np.minimum(delta, max_value - offset)

    #     return delta, offset

    def gemmlowpClippingQuantize(self, tensor, id, tag="", clip_type='laplace'):
        
        if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):  # conv的输出且非(x,x,1,1)这种类型
            stats = self.__act_stats_perchannel__(tensor, ['min', 'max'], avg_over_batch=False)
            mean = self.__act_stats_perchannel__(tensor, ['mean'], avg_over_batch=True)['mean'] # 这里有没有avg_over_batch好像都一样？
        else:
            stats = self.__act_stats__(tensor, ['min', 'max', 'mean'], avg_over_batch=False)
            mean = stats['mean']
        min_value = stats['min']  #(64,)
        max_value = stats['max']  #(64,)
        # mean = stats['mean']

        if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1) \
                and len(min_value) > 1 and len(max_value) > 1: # 进这个perchannel
            alpha, mask = self.get_alpha_laplace(tensor, mean, per_channel=True)  # ok, aciq一定是对perchannel操作的。
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean, mask)
            min_value = to_cuda(min_value, tensor.device)
            range = to_cuda(range, tensor.device)

            tnew = tensor.contiguous()
            N, C, H, W = tnew.shape  # N x C x H x W
            t = tnew.detach().transpose(0, 1).contiguous()  # C x N x H x W
            t = t.view(t.shape[0], -1)
            output = self.__gemmlowpQuantize1__(t, range, min_value)
            output = output.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
            res =  output.view(tnew.shape)
        else:
            alpha, mask = self.get_alpha_laplace(tensor, mean, per_channel=False)
            max_value = float(max_value); min_value = float(min_value); mean = float(mean); alpha = float(alpha)
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean, mask)
            res = self.__gemmlowpQuantize1__(tensor.contiguous(), to_cuda(range, tensor.device), to_cuda(min_value, tensor.device))

        return res

    def gemmlowpMinMaxQuantize(self, tensor, tag=""):
        stats = self.__act_stats__(tensor, ['min', 'max'], avg_over_batch=('activation' in tag and 'classifier' not in tag)) # act_pool和act_classi用这个，所以 act_pool 用 avg_over_batch；而act_classi（最后一层activation）不用。
        min_ = stats['min']
        max_ = stats['max']

        if self.half_range:
            min_ = 0

        return self.__gemmlowpQuantize1__(tensor, max_ - min_, min_)

    def gemmlowpQuantizeActivationPerChannel(self, tensor, id, tag="", min_=None, max_=None):
        if min_ is None:
            if self.half_range:
                min_ = 0  # np.zeros(min_.shape)
            else:
                min_ = self.__act_stats_perchannel__(tensor, ['min'], avg_over_batch=False)['min']
        min_ = to_cuda(min_, tensor.device)

        if max_ is None:
            max_ = self.__act_stats_perchannel__(tensor, ['max'], avg_over_batch=False)['max']
        max_ = to_cuda(max_, tensor.device)

        N, C, H, W = tensor.shape  # N x C x H x W
        t = tensor.detach().transpose(0, 1).contiguous()  # C x N x H x W
        t = t.view(t.shape[0], -1)

        output = self.__gemmlowpQuantize1__(t, max_ - min_, min_)

        output = output.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
        return output.view(tensor.shape)

    def gemmlowpQuantizeWeightsPerChannel(self, tensor, id, min_=None, max_=None):
        # Assume weights with dimensions [OFM,IFM,K1,K2]
        t = tensor.view(tensor.shape[0], -1)  #tensor(64,3,7,7) t(64,147)    #linear：（1000，2048）

        # per output channel min, max
        if min_ is None:
            min_ = t.min(-1)[0]    #(64,)
        if max_ is None:
            max_ = t.max(-1)[0]   #

        output = self.__gemmlowpQuantize1__(t, max_ - min_, min_)  #(64,147)

        return output.view(tensor.shape)  #torch.Size([64, 3, 7, 7])

    @staticmethod
    def __act_stats__(tensor, stats, avg_over_batch=False):
        # Assume activation dimentions [N,C,H,W]
        # avg_over_batch: true的话对每个instance求min max，对一个batchsize的min max求平均作为最后的minmax；false的话就是对batch中的所有数求min max
        t = tensor.view(tensor.shape[0], -1) if avg_over_batch else tensor.view(-1) # [N, CxHxW] or [NxCxHxW]

        stats_dict = {}
        for s in stats:
            if s == 'max':
                stats_dict[s] = t.max(dim=-1)[0] if avg_over_batch else t.max()
            elif s == 'min':
                stats_dict[s] = t.min(dim=-1)[0] if avg_over_batch else t.min()
            elif s == 'mean':
                stats_dict[s] = t.mean(dim=-1) if avg_over_batch else t.mean()
            elif s == 'b': # 与mean的偏移值，# laplace的b这样算吗？确实
                stats_dict[s] = torch.mean(torch.abs(t - t.mean(dim=-1).unsqueeze(-1)), dim=-1) if avg_over_batch else torch.mean(torch.abs(t - t.mean()))
            elif s == 'std': # 标准差
                stats_dict[s] = torch.std(t, dim=-1, unbiased=True) if avg_over_batch else t.std(unbiased=True)

            if avg_over_batch:
                stats_dict[s] = torch.mean(stats_dict[s], dim=0)

        return stats_dict

    @staticmethod
    def __act_stats_perchannel__(tensor, stats, avg_over_batch=False):
        # Assume activation dimentions [N,C,H,W]
        if not avg_over_batch: #
            t = tensor.transpose(0, 1).contiguous()  # [C, N, H, W]
            t = t.view(t.shape[0], -1) # [C, NxHxW]
        else:
            t = tensor.view(tensor.shape[0], tensor.shape[1], -1)  # [N, C, HxW]

        stats_dict = {}
        for s in stats:
            if s == 'max':
                stats_dict[s] = t.max(dim=-1)[0]
            elif s == 'min':
                stats_dict[s] = t.min(dim=-1)[0]
            elif s == 'mean':
                stats_dict[s] = t.mean(dim=-1)
            elif s == 'b':
                stats_dict[s] = torch.mean(torch.abs(t - t.mean(dim=-1).unsqueeze(-1)), dim=-1) # laplace的b这样算吗？确实
            elif s == 'std':
                stats_dict[s] = torch.std(t, dim=-1, unbiased=True)

            if avg_over_batch:
                stats_dict[s] = torch.mean(stats_dict[s], dim=0)

        return stats_dict

    def __gemmlowpQuantize1__(self, tensor, delta, offset):
        qmin = 0.     #delta= weight 的 max_ - min_, offset是min_
        qmax = 2.**self.num_bits - 1.   #255
        scale = (delta) / (qmax - qmin)  #

        scale = torch.max(scale, torch.tensor([1e-8]).to(tensor.device))

        output = tensor.detach()
        if self.enforce_true_zero: #
            initial_zero_point = qmin - offset / scale
            # make zero exactly represented
            zero_point = torch.round(initial_zero_point)
            output = torch.div(output, scale.unsqueeze(-1))   #  x/s + z
            output = torch.add(output, zero_point.unsqueeze(-1))
        else:
            output = torch.add(output, -offset.unsqueeze(-1))
            output = torch.div(output, scale.unsqueeze(-1))

        output.clamp_(qmin, qmax).round_()  # quantize     #做clamp 操作 得到xint  #因为是模拟，还是torch.float32

        if self.enforce_true_zero:   #
            output = torch.add(output, -zero_point.unsqueeze(-1))
            output = torch.mul(output, scale.unsqueeze(-1))  # dequantize    对 变成x hat
        
        else:
            output = torch.mul(output, scale.unsqueeze(-1))
            output = torch.add(output, offset.unsqueeze(-1))  # dequantize

        # workaround for out of memory issue
        torch.cuda.empty_cache()

        return output.view(tensor.shape)

def int_quantizer(qtype, quant_params):

    size = int(qtype[len('int'):]) # int8->8!

    return IntQuantizer(size, quant_params)

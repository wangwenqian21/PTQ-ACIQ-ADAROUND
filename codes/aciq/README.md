## 文件说明
我们主要参考了 https://github.com/submission2019/cnn-quantization 的代码实现，我们阅读了他们的代码后做了一些修剪，保留了aciq比较重要的部分和一些aciq中提及有助于量化的性能的技巧。
下面主要说明几个核心文件：

- /home/group6/final_version/aciq/pytorch_quantizer/quantization/qtypes/int_quantizer.py 中实现了对weight和activation进行量化的quantizer。
- /home/group6/final_version/aciq/pytorch_quantizer/quantization/inference/inference_quantization_manager.py 中实现了换层的开关，以及换层类和它的forward函数。它调用了int_quantizer.py中实现的int_quantizer，通过设置不同的参数得到了用于量化不同层、不同对象的quantizer。
- ./utils/absorb_bn.py 中实现了bn folding功能，
- ./utils/mark_relu.py 中实现了对relu层前的层进行标记，用于决定是否近优化[0,\alpha]区间

## 对于aciq源码的主要修改：

- 增加对convactivation relu前的conv的mask
- 对frozenbn也做bn_folding
- optimization_experiment 文件夹下是我们对fused_relu优化问题做的一些尝试。这个尝试我们在 /home/group6/final_version/aciq/pytorch_quantizer/quantization/qtypes/int_quantizer.py 中也有实现，由于效果不佳注释掉了。


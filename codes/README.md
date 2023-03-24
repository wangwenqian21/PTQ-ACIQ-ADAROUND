# 文件结构

- aciq：aciq任务代码。参考并修改了aciq的源码。
- adaround：adaround任务代码。参考并修改了https://github.com/Ranking666/Base-quantization的实现。
- baseline：我们做的baseline实验，即无量化时的效果。

具体文件结构见每个文件夹内的readme文件。一些通用的文件被放进了各文件夹的util或utils文件夹，它们有统一的名字如下：

- iou.py：deeplabv3的miou metric。参考了 https://github.com/VainF/DeepLabV3Plus-Pytorch 中的miou metric实现。
- voc.py：deeplabv3的数据载入实现，参考了csdn上的代码实现
- ext_transforms.py：deeplabv3的数据变换，参考了 https://github.com/VainF/DeepLabV3Plus-Pytorch 中的数据变换实现。

# 关于环境

实验环境在`baseline.yaml`中。需要的库主要有 pytorch, torchvision, pycocotools。

此外关于aciq部分我们改了一处库函数。
retinanet的convactivation层由于nn.relu在默认参数上，无法在aciq的换层中被替换，需要将该nn.relu写在函数体里才能够替换。

```
# 从aciq/utils/mark_relu.py中找到Conv2dNormActivation类的库文件位置，152行开始，即在super().__init__前，增加如下的
if str(activation_layer)=="<class 'torch.nn.modules.activation.ReLU'>":
    activation_layer = torch.nn.ReLU
```

# 运行指令

## Baseline
```
cd baseline
conda activate baseline
CUDA_VISIBLE_DEVICES=0 python resnet_orgin.py
CUDA_VISIBLE_DEVICES=0 python retinanet_orgin.py
CUDA_VISIBLE_DEVICES=0 python deeplabv3_orgin.py
```

## ACIQ

```
cd aciq
conda activate aciq
```

### ResNet50

```
bash run_resnet.sh
```

### RetinaNet

```
bash run_retinanet.sh
```

### DeepLabv3

```
bash run_dlv3.sh
```

## AdaRound

```
cd AdaRound
conda activate baseline
```
其中train模式是从头训练adaround，infer模式是加载训练好的ckpt直接用于推理。

### ResNet50
```
CUDA_VISIBLE_DEVICES=0 python resnet.py --mode train
CUDA_VISIBLE_DEVICES=0 python resnet.py --mode infer
```

### RetinaNet
```
CUDA_VISIBLE_DEVICES=0 python retinanet.py --mode train
CUDA_VISIBLE_DEVICES=0 python retinanet.py --mode infer
```

### DeepLabv3
```
CUDA_VISIBLE_DEVICES=0 python deeplabv3.py --mode train
CUDA_VISIBLE_DEVICES=0 python deeplabv3.py --mode infer
```
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn as nn

def mark_bottlenetck_before_relu(model):
    for m in model.children():
        if isinstance(m, Bottleneck):
            m.conv1.before_relu = True
            m.bn1.before_relu = True
            m.conv2.before_relu = True
            m.bn2.before_relu = True
        else:
            mark_bottlenetck_before_relu(m)

def mark_basicblock_before_relu(model):
    for m in model.children():
        if isinstance(m, BasicBlock):
            m.conv1.before_relu = True
            m.bn1.before_relu = True
        else:
            mark_basicblock_before_relu(m)

def mark_cna_before_relu(model):
    for m in model.children():
        if isinstance(m, Conv2dNormActivation):
            # import pdb
            # pdb.set_trace()
            for _,mm in m.named_modules():
                if isinstance(mm, nn.Conv2d):
                    mm.before_relu = True
        else:
            mark_cna_before_relu(m)

def resnet_mark_before_relu(model):
    model.conv1.before_relu = True

    mark_bottlenetck_before_relu(model)
    mark_basicblock_before_relu(model)

def deeplabv3_mark_before_relu(model):
    model.backbone.conv1.before_relu = True

    mark_bottlenetck_before_relu(model)
    mark_basicblock_before_relu(model)

def retinanet_mark_before_relu(model):
    model.backbone.body.conv1.before_relu = True
    mark_cna_before_relu(model.head)
    mark_bottlenetck_before_relu(model)
    mark_basicblock_before_relu(model)

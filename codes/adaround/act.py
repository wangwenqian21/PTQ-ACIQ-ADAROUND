from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn as nn

def mark_bottlenetck_before_relu(model, actlist, name):
    for n,m in model.named_children():
        if isinstance(m, Bottleneck):
            actlist[name+'.'+n+'.conv1'] = m.relu
            actlist[name+'.'+n+'.conv2'] = m.relu
        else:
            mark_bottlenetck_before_relu(m, actlist, name + '.' + n)

def mark_basicblock_before_relu(model, actlist, name):
    for n,m in model.named_children():
        if isinstance(m, BasicBlock):
            actlist[name+'.'+n+'.conv1'] = m.relu
        else:
            mark_basicblock_before_relu(m, actlist, name + '.' + n)
relu = nn.ReLU()
def mark_cna_before_relu(model, actlist, name):
    for n,m in model.named_children():
        if isinstance(m, Conv2dNormActivation):
            for small_name,mm in m.named_modules():
                if isinstance(mm, nn.Conv2d):
                    actlist[name+'.'+n + '.' +small_name] = relu
        else:
            mark_cna_before_relu(m, actlist, name + '.' + n)

def resnet_mark_before_relu(model):
    name = ''
    actlist = {'conv1':model.relu}
    mark_bottlenetck_before_relu(model, actlist, name)

    return actlist

def deeplabv3_mark_before_relu(model):
    name = ''
    actlist = {'backbone.conv1':model.backbone.relu}
    mark_bottlenetck_before_relu(model, actlist, name)

    return actlist

def retinanet_mark_before_relu(model):
    name = ''
    actlist = {'backbone.body.conv1':model.backbone.body.relu}
    mark_cna_before_relu(model.head, actlist, name+'.head')
    mark_bottlenetck_before_relu(model, actlist, name)
    mark_basicblock_before_relu(model, actlist, name)
    return actlist
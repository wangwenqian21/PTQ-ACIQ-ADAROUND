import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm

import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
#print(sys.path)
import util.ext_transforms as et
from util.iou import StreamSegMetrics 


def evaluate():  
  with torch.no_grad():
    model.eval()
    for images, labels in tqdm(val_dataloader, desc="Processing"):
      images = images.cuda()
      labels = labels.cuda()
      outputs = model(images)['out']
      preds = outputs.detach().max(dim=1)[1].cpu().numpy() 
      targets = labels.cpu().numpy() 

      metrics.update(targets, preds) 
    score = metrics.get_results()
    print(metrics.to_str(score))

  
val_transform = et.ExtCompose([
    et.ExtResize(520),
    et.ExtCenterCrop(520),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])
val_data = VOCSegmentation(root="/home/cs433/files/benchmark/VOC2012/", year='2012',
                            image_set='val', download=False, transforms=val_transform)


val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT)
model = model.cuda()

metrics = StreamSegMetrics(21)

evaluate()


"""
Processing: 100%|██████████████████████████████████████████████████| 1449/1449 [01:30<00:00, 15.99it/s]

Overall Acc: 0.943965
Mean Acc: 0.871824
FreqW Acc: 0.899572
Mean IoU: 0.778310
"""
import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet50_Weights


def evaluate():  #val_dataloader 快
    net.eval()
    numall = 0
    acc = 0
    top5acc = 0
    with torch.no_grad():
        for i,(inputs,targets) in tqdm(enumerate(val_dataloader), total=len(val_dataloader),desc="evaluating"):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')   #print(targets.shape)  #torch.Size([64])
            outputs = net(inputs)   #print(outputs.shape)  #torch.Size([64, 1000])  
   
            sorted,indices = torch.sort(outputs, dim = -1, descending = True)  #重新排列后的tensor，和index  #print(sorted.shape,indices.shape)  #torch.Size([64, 1000]) torch.Size([64, 1000])
           
            for i in range(inputs.shape[0]):
                top5 = indices[i,:5].cpu().numpy().tolist()
                label = targets[i].item()   #label
                if label in top5:
                    top5acc += 1
                    if label == top5[0]:
                        acc +=1
                numall += 1

    print('top1 acc: ', acc / numall)
    print('top5 acc: ', top5acc / numall)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_data = datasets.ImageFolder(root = "/home/cs433/files/benchmark/ImageNet/val", transform=ResNet50_Weights.IMAGENET1K_V1.transforms())
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)

net = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
net = net.to(device)    

evaluate()

    

# 12.22
# evaluating: 100%|████████████████████████████████████████████████████████████████████████| 782/782 [10:15<00:00,  1.27it/s]
# top1 acc:  0.7613
# top5 acc:  0.92862


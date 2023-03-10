import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from util.iou import StreamSegMetrics 
import util.ext_transforms as et 
from torchvision.datasets import VOCSegmentation 
import argparse
from utils import inplace_quantize_layers, enable_calibrate, disable_calibrate, calibrate_adaround, setmylog, adaround_infer
HOLE = 32
BS = 8
NB = int(HOLE/BS)
def calibrate():   
    net.eval()
    with torch.no_grad():
        for i,(inputs,targets) in tqdm(enumerate(val_dataloader),total=len(val_dataloader), desc="calibrating"):    #torch.Size([1, 3, 520, 520]),  torch.Size([1, 520, 520])
            if i==NB: # 原文1280；时间不够，128张好了(2个batch)
                break
            res = net.forward(inputs.cuda())


def evaluate(): 
    net.eval()
    with torch.no_grad():
        for i,(inputs,targets) in tqdm(enumerate(val_dataloader),total=len(val_dataloader), desc="evaluating"): 
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = net(inputs)['out']
            preds = outputs.detach().max(dim=1)[1].cpu().numpy() 
            targets = targets.cpu().numpy()
            metrics.update(targets, preds) 
            
        score = metrics.get_results() ###### 用这个 ######
        logger.info(metrics.to_str(score))


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Q')
    parser.add_argument('--load_ckpt', action='store_true', help="断点续训（从某层开始）")
    parser.add_argument('--mode',default='infer',choices=['infer','train'],help="infer or train")
    args = parser.parse_args()
    infer = args.mode
    model_name = 'deeplabv3'
    logger = setmylog(model_name)
    logger.info(str(args))
    
    # Config  #PTQ
    b_start=20
    b_end=2
    warmup=0.2
    adaround_iter=1000
    ptq=True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('==> Preparing data..')    
    val_transform = et.ExtCompose([
    et.ExtResize(520),
    et.ExtCenterCrop(520),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),])
    
    #data_dir="/data/ygr/benchmark/VOC2012/"
    # data_dir= "/home/ubuntu/multi/benchmark/VOC2012/"
    data_dir= "/home/cs433/files/benchmark/VOC2012/"
    val_data = VOCSegmentation(root=data_dir, year='2012',image_set='val', download=False, transforms=val_transform)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    fake_traindata = VOCSegmentation(root=data_dir, year='2012',image_set='val', download=False, transforms=val_transform)
    fake_traindataloader = DataLoader(fake_traindata, batch_size=BS, shuffle=False, num_workers=0)

    # logger.info(len(val_data))  #1449
    # logger.info(len(fake_traindata))   #1449
    # logger.info(len(val_dataloader))    #1449
    # logger.info(len(fake_traindataloader))  #1449

    logger.info('==> Building model..')
    net = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT)
    net = net.to(device)    
    metrics = StreamSegMetrics(21)

    logger.info("==> inplace_quantize_layers..")
    net = inplace_quantize_layers(net,ptq) #469*20   
    net = net.to(device)  #插入了quantize层

    logger.info("==> calibrate...")
    enable_calibrate(net)   #递归地都改成true
    calibrate()  # 插入fake的 evaluate   这一步的时候scale 和zero point 已经确定了 关掉了ptq
    disable_calibrate(net)
    
    logger.info("==> adaround...")
    if infer=='infer': #load ckpt
        adaround_infer(net, model_name)
    else:
        loader = []
        for idx, (batch, _) in enumerate(fake_traindataloader):
            if idx ==NB: # 128张好了
                break
            loader.append(batch)

        calibrate_adaround(model_name, net, adaround_iter, b_start, b_end, warmup, loader, device, logger, args.load_ckpt)  # apply adaraound   #这里不在做observer和 update_qparams 只更新h
    
    net = net.to(device)
    logger.info("==> evaluate...")
    evaluate()


# python deeplabv3.py
# nohup python deeplabv3.py > deeplabv3.log &



# evaluating: 100%|██████████████████████████████████████████████████| 1449/1449 [02:20<00:00, 10.30it/s]

# Overall Acc: 0.943432
# Mean Acc: 0.872373
# FreqW Acc: 0.898920
# Mean IoU: 0.777586



#adaround:  49%|███████▊        | 492/1000 [00:37<00:36, 13.99it/s, recon_loss=0.0619, round_loss=1.65e+3]


# python deeplabv3.py --mode train
# python deeplabv3.py --mode infer
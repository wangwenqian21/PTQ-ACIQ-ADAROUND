import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
from utils import inplace_quantize_layers, enable_calibrate, disable_calibrate, calibrate_adaround, setmylog, adaround_infer
import argparse
HOLE = 128
BS = 128
num_batch = HOLE/BS
# print(num_batch==1)
# exit()
def calibrate(): 
    net.eval()
    with torch.no_grad():                      
        for i,(inputs,targets) in tqdm(enumerate(fake_traindataloader), total=len(fake_traindataloader), desc="calibrating"):
            if i==num_batch:   #calibrate只需要少量图片 (1280张图片) # 128张图好了(1个batch)
                break
            inputs = inputs.to(device)
            outputs = net(inputs)   #logger.info(outputs.shape)  #torch.Size([128, 1000])   #logger.info(res.shape)  # (1,1000)
           

def evaluate():  #val_dataloader 快
    net.eval()
    numall = 0
    acc = 0
    top5acc = 0
    with torch.no_grad():
        for i,(inputs,targets) in tqdm(enumerate(val_dataloader), total=len(val_dataloader),desc="evaluating"):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')   #logger.info(targets.shape)  #torch.Size([64])
            outputs = net(inputs)   #logger.info(outputs.shape)  #torch.Size([64, 1000])  
   
            sorted,indices = torch.sort(outputs, dim = -1, descending = True)  #重新排列后的tensor，和index  #logger.info(sorted.shape,indices.shape)  #torch.Size([64, 1000]) torch.Size([64, 1000])
           
            for i in range(inputs.shape[0]):
                top5 = indices[i,:5].cpu().numpy().tolist()
                label = targets[i].item()   #label
                if label in top5:
                    top5acc += 1
                    if label == top5[0]:
                        acc +=1
                numall += 1

    logger.info('top1 acc: {}'.format(acc / numall))
    logger.info('top5 acc: {}'.format(top5acc / numall))
    
    
if __name__ == "__main__":
    # Config  #PTQ
    parser = argparse.ArgumentParser(description='Q')
    #default='L', choices=['L','C'],  help='per_channel or per_tensor')
    parser.add_argument('--load_ckpt', action='store_true', help="断点续训（从某层开始）")
    parser.add_argument('--mode',default='infer',choices=['infer','train'],help="infer or train")
    args = parser.parse_args()
    infer = args.mode
    model_name = 'resnet'
    logger = setmylog(model_name)
    logger.info(str(args))
    
    b_start=20
    b_end=2
    warmup=0.2
    adaround_iter=1000
    ptq=True
    args = parser.parse_args()
    infer=args.mode
    # infer=True
    # infer=False
    
    #load_ckpt = '/home/ubuntu/multi/wwq_adaround/round_wwq.pth' # None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('==> Preparing data..')
    # data_dir = "/home/ubuntu/multi/benchmark/ImageNet/val"
    # data_dir = '/data/ygr/benchmark/ImageNet/val'
    data_dir="/home/cs433/files/benchmark/ImageNet/val"
    val_data = datasets.ImageFolder(root = data_dir, transform=ResNet50_Weights.IMAGENET1K_V1.transforms())
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    fake_traindata = datasets.ImageFolder(root = data_dir, transform=ResNet50_Weights.IMAGENET1K_V1.transforms())
    fake_traindataloader = DataLoader(fake_traindata, batch_size=BS, shuffle=False, num_workers=0)
    
    logger.info('==> Building model..')
    net = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
    # import pdb
    # pdb.set_trace()
    net = net.to(device)    
    
    logger.info("==> inplace_quantize_layers..")
    net = inplace_quantize_layers(net,ptq) #469*20   #check OK
    net = net.to(device)  #插入了quantize层
    
    logger.info("==> calibrate...")
    enable_calibrate(net)   #递归地都改成true
    calibrate()   # 插入fake的 evaluate   这一步的时候scale 和zero point 已经确定了 关掉了ptq
    disable_calibrate(net)
    

        
    logger.info("==> adaround...")
    if infer=='infer':  #load ckpt
        adaround_infer(net, model_name)
    else:  # train from scratch
        loader = []
        for idx, (batch, _) in enumerate(fake_traindataloader):
            if idx ==num_batch: # 128张图好了
                break
        loader.append(batch)
        calibrate_adaround(model_name, net, adaround_iter, b_start, b_end, warmup, loader, device, logger, args.load_ckpt)    # apply adaraound   #这里不在做observer和 update_qparams 只更新h
    
    net.to(device)
    logger.info("==> evaluate...")
    evaluate()


# 指令 
# python resnet.py 
# nohup python resnet.py > resnet.log &

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lustre/sjtu/home/gry10/anaconda3/envs/adaround/lib

# python resnet.py --mode train
# python resnet.py --mode infer



# Overall Acc: 0.943432
# Mean Acc: 0.872373
# FreqW Acc: 0.898920
# Mean IoU: 0.777586
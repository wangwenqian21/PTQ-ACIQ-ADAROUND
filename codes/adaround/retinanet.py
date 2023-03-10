import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

import torchvision.datasets as dset
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from tqdm import tqdm
import json
from utils import inplace_quantize_layers, enable_calibrate, disable_calibrate, calibrate_adaround, disable_origin, setmylog, adaround_infer
import argparse

HOLE = 10
BS = 1

def box_trans(bbox_list):
    """
    input model output format (x left_bottom ,y left_bottom , x right top,y right top)
    output [x left top, y left top, width ,height]
    """
    
    xmin,ymin,xmax,ymax=bbox_list[0],bbox_list[1],bbox_list[2],bbox_list[3]
    x_left_top=xmin
    y_left_top=ymin  # 图片的坐标原点在左上角，所以left top 是y min 服了!!!!
    width=xmax-xmin
    height=ymax-ymin
    
    return [x_left_top,y_left_top,width,height]   


def calibrate():
    net.eval()
    with torch.no_grad(): 
        for i,(img,label) in tqdm(enumerate(coco_val),total=len(coco_val), desc="calibrating"):  #len(coco_val)) 5000   #img <PIL.Image.Image image mode=RGB size=640x426 at 0x7F7FBEB62D30>     #logger.info(img.size)   #(640,426) width*height   
            if i==1280:  #calibrate只需要少量图片 (1280张图片)
                break
            try:
                image_id = label[0]['image_id']
            except:
                continue   # logger.info(label)  有一些label是[]

            # tf=T.ToTensor()  #这样就可以
            # transformed = tf(img)   #torch.Size([3, 426, 640])  #(height*width)  
            batched = img.unsqueeze(0) 
            batched = batched.to('cuda:0')
            # import pdb
            # pdb.set_trace()
            output = net(batched)[0]      # 输入要求 a list of tensors, each of shape [C, H, W]


def evaluate(): 
    net.eval()
    results = []
    
    with torch.no_grad(): 
        for i,(img,label) in tqdm(enumerate(coco_val),total=len(coco_val),desc="evaluating"):  #len(coco_val)) 5000
            try:
                image_id = label[0]['image_id']
            except:
                continue   # logger.info(label)  有一些label是[]

            # tf=T.ToTensor()  #这样就可以
            # transformed = tf(img)   #torch.Size([3, 426, 640])  #(height*width)  
            batched = img.unsqueeze(0) 
            batched = batched.to('cuda:0')
    
            output = net(batched)[0]      # 输入要求 a list of tensors, each of shape [C, H, W]
            boxes=output['boxes'] 
            scores=output['scores'] 
            labels=output['labels']    #logger.info(boxes.shape,scores.shape,labels.shape)  #torch.Size([241, 4]) torch.Size([241]) torch.Size([241])

 
            threshold=0  # We will use the threshold parameter to select bounding boxes with a certain level of confidence or higher. The rule of thumb is to use 0.5
            
            num=labels.shape[0]   
            for i in range(num):    
                score=scores[i].item()
                if score < threshold:  # scores are sorted, so we can break  
                    break
                box= boxes[i].tolist() 
                box= box_trans(box)    # retinanet output bbox 和 coco dataset 的格式不一样，需要转换
                        
                results.append({"image_id": image_id,  
                                "category_id": labels[i].item(),
                                "bbox":box,
                                "score": score})
        
            
    with open(c2json, "w") as f:
            json.dump(results, f, indent=4)

    cocoAnnotation = COCO(annotation_file=path2json)  #ground_truth
    cocovalPrediction = cocoAnnotation.loadRes(c2json)
    cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q')
    #default='L', choices=['L','C'],  help='per_channel or per_tensor')
    parser.add_argument('--load_ckpt', action='store_true', help="断点续训（从某层开始）")
    parser.add_argument('--mode',default='infer',choices=['infer','train'],help="infer or train")
    args = parser.parse_args()
    infer = args.mode
    model_name = 'retinanet'
    logger = setmylog(model_name)
    logger.info(str(args))
    
    b_start=20
    b_end=2
    warmup=0.2
    adaround_iter=1005
    ptq=True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    logger.info('==> Preparing data..')
    
    
    # path2data="/home/ubuntu/multi/benchmark/COCO/val2017"
    # path2json="/home/ubuntu/multi/benchmark/COCO/annotations/instances_val2017.json"
    path2data="/home/cs433/files/benchmark/COCO/val2017"
    path2json="/home/cs433/files/benchmark/COCO/annotations/instances_val2017.json"
    # c1json="c1.json"
    c2json="c2.json"
    
    coco_val = dset.CocoDetection(root = path2data,annFile = path2json, transform=RetinaNet_ResNet50_FPN_Weights.COCO_V1.transforms())   #搞不了batch
    # logger.info(str(len(coco_val)))  #5000

    logger.info('==> Building model..')
    net = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)  # or (pretrained=True)
    net = net.to(device)    
    
    logger.info("==> inplace_quantize_layers..")
    net = inplace_quantize_layers(net, ptq)
    net = net.to(device)  #插入了quantize层
    # logger.info(net)
    # exit(0)

    logger.info("==> calibrate...")    
    enable_calibrate(net)  
    calibrate()  
    disable_calibrate(net)
    
    logger.info("==> adaround...")
    if infer=='infer':  #load ckpt
        adaround_infer(net,model_name)
    else:
        loader = []
        abatch = []
        for i,(img,_) in tqdm(enumerate(coco_val),total=len(coco_val),desc="evaluating"):  #len(coco_val)) 5000
            if i==HOLE:
                break
            if (i+1)%BS==0:
                loader.append(abatch[:])
                abatch = []
            # tf=T.ToTensor()  #这样就可以
            # transformed = tf(img)   #torch.Size([3, 426, 640])  #(height*width)  
            # batched = batched.to('cuda:0')
            abatch.append(img)
        calibrate_adaround(model_name, net, adaround_iter, b_start, b_end, warmup, loader, device, logger, args.load_ckpt)  
    
    disable_origin(net)
    net = net.to(device)
    logger.info("==> evaluate...")
    evaluate()


# 指令 python retinanet.py 
# nohup python retinanet.py  > retinanet.log &


#adaround:  26%|████▍            | 261/1005 [00:41<01:46,  7.02it/s, recon_loss=0.183, round_loss=2.46e+3]

# python retinanet.py --mode train
# python retinanet.py --mode infer



#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.488
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
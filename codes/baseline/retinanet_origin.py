import torchvision
import torch
import torchvision.datasets as dset
from torchvision import transforms as T
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from tqdm import tqdm
import json

def box_trans(bbox_list):
    # input model output format (x left_bottom ,y left_bottom , x right top,y right top)
    # output [x left top, y left top, width ,height]
    xmin,ymin,xmax,ymax=bbox_list[0],bbox_list[1],bbox_list[2],bbox_list[3]
    
    x_left_top=xmin
    y_left_top=ymin  # 图片的坐标原点在左上角，所以left top 是y min 服了!!!!
    width=xmax-xmin
    height=ymax-ymin
    return [x_left_top,y_left_top,width,height]   

    
def main():    
    # path2data="/home/ubuntu/multi/benchmark/COCO/val2017"
    # path2json="/home/ubuntu/multi/benchmark/COCO/annotations/instances_val2017.json"
    path2data="/home/cs433/files/benchmark/COCO/val2017"
    path2json="/home/cs433/files/benchmark/COCO/annotations/instances_val2017.json"
    pred2json="val_pred.json"
    coco_val = dset.CocoDetection(root = path2data,annFile = path2json, transform=RetinaNet_ResNet50_FPN_Weights.COCO_V1.transforms())

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)  # or (pretrained=True)
    model=model.cuda()

    model.eval()
    results = []
    print(model)
    with torch.no_grad():  #speeds up 
        for i,(img,label) in tqdm(enumerate(coco_val),total=len(coco_val),desc="evaluating"):    #len(coco_val)) 5000
        #len(coco_val)) 5000   #img <PIL.Image.Image image mode=RGB size=640x426 at 0x7F7FBEB62D30>         #print(img.size)   #(640,426) width*height     
            try:
                image_id = label[0]['image_id']
            except:
                continue   # print(label)  有一些label是[]
                   
            # tf=T.ToTensor()  #这样就可以
            # import pdb
            # pdb.set_trace()
            # continue
            # transformed = tf(img)   #torch.Size([3, 426, 640])  #(height*width)  
            batched = img.unsqueeze(0) 
            batched = batched.to('cuda:0')
    
            output = model(batched)[0]      # 输入要求 a list of tensors, each of shape [C, H, W]
            boxes=output['boxes'] 
            scores=output['scores'] 
            labels=output['labels']
            #print(boxes.shape,scores.shape,labels.shape)  #torch.Size([241, 4]) torch.Size([241]) torch.Size([241])
 
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
            
    with open(pred2json, "w") as f:
            json.dump(results, f, indent=4)
    
    cocoAnnotation = COCO(annotation_file=path2json)  #ground_truth
    cocovalPrediction = cocoAnnotation.loadRes(pred2json)
    cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    #label:
    #{'segmentation': [[240.86, 211.31, 240.16, 197.19, 236.98, 192.26, 237.34, 187.67, 245.8, 188.02, 243.33, 176.02, 250.39, 186.96, 251.8, 166.85, 255.33, 142.51, 253.21, 190.49, 261.68, 183.08, 258.86, 191.2, 260.98, 206.37, 254.63, 199.66, 252.51, 201.78, 251.8, 212.01]],
    # 'area': 531.8071000000001,
    # 'iscrowd': 0, 
    # 'image_id': 139, 
    # 'bbox': [236.98, 142.51, 24.7, 69.5], 
    # 'category_id': 64, 
    # 'id': 26547}

if __name__ == '__main__':
    main()
    

"""
(baseline) group6@38f495981d79:~/adaround/my_adaround$ /home/group6/miniconda3/envs/baseline/bin/python /home/group6/adaround/my_adaround/origin/retinanet_origin.py
loading annotations into memory...
Done (t=0.72s)
creating index...
index created!
evaluating: 100%|███████████████████████████████████████████████████████████████████| 5000/5000 [05:26<00:00, 15.30it/s]
loading annotations into memory...
Done (t=0.55s)
creating index...
index created!
Loading and preparing results...
DONE (t=6.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=59.91s).
Accumulating evaluation results...
DONE (t=17.33s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.557
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.696
 """
 
 
 
# 换上dataloader 也就这速度
# 100%|██████████████████████████████████████████████████████████| 5000/5000 [06:59<00:00, 11.92it/s]
# 图片就是有不同尺寸的 而且对应不同metric 不能裁成一样大小
# 而且有些label 是空的 也没法做
# 我感觉就是没办法直接变成batch>1 的dataloader

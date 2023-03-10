
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(dir_path, os.path.pardir)
sys.path.append(root_dir)
from tqdm import tqdm

import argparse
import json
import logging
import random
import time
import warnings
import torch
import torch.backends.cudnn as cudnn
# import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from pytorch_quantizer.quantization.inference.inference_quantization_manager import QuantizationManagerInference as QM
from utils.meters import AverageMeter, accuracy, StreamSegMetrics 
from utils.absorb_bn import search_absorbe_bn
from utils.mark_relu import resnet_mark_before_relu, deeplabv3_mark_before_relu, retinanet_mark_before_relu
from pathlib import Path

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
torch.backends.cudnn.deterministic = True
'''
torch.backends.cudnn.benchmark = True，在训练的时候，一个好的特征可能会经过好多不同的convolution algorithms的测试，选出来最快的，剩下的过程都会采用这个算法。
torch.backends.cudnn.deterministic = True 只会针对这一个算法进行控制，而torch.use_deterministic_algorithms(True) 会对其他的操作产生影响
'''

home = str(Path.home())
# IMAGENET_FOR_INFERENCE = '/home/ubuntu/multi/benchmark/ImageNet/'
IMAGENET_FOR_INFERENCE = '/home/cs433/files/benchmark/ImageNet/'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default=IMAGENET_FOR_INFERENCE,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    choices=['resnet50','retinanet', 'deeplabv3'])
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='logger.info frequency (default: 10)') # 几轮打印一次
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')

parser.add_argument('--qtype', default=None, help='data type: bfloat[N], int[N]')   # acti quantize成int
parser.add_argument('--qweight', '-qw', default='int8', help='quantizer for weights')  # weight quantize成多少
parser.add_argument('--shuffle', '-sh', action='store_true', help='shuffle data') # -sh是shuffle data
parser.add_argument('--q_off', action='store_true', help='disable quantization') # -sh是shuffle data
parser.add_argument('--preserve_zero', '-pz', action='store_true', help='Preserve zero during quantization', default=False) # 0还是0
parser.add_argument('--clipping', '-c', default='no', help='Clipping type: [no, laplace]')   #laplace  ACIQ
parser.add_argument('--per_channel_quant_weights', '-pcq_w', action='store_true', help='Per channel quantization of weights', default=False)  #
parser.add_argument('--per_channel_quant_act', '-pcq_a', action='store_true', help='Per channel quantization of activations', default=False)  #
parser.add_argument('--bias_corr_act', '-bca', action='store_true', help='Bias correction for activations', default=False)
parser.add_argument('--bias_corr_weight', '-bcw', action='store_true', help='Bias correction for weights', default=False)   ## Bias correction
parser.add_argument('--var_corr_weight', '-vcw', action='store_true', help='Variance correction for weights', default=False)   ## Bias correction
parser.add_argument('--fused_relu', '-fr', action='store_true', help='Frsed relu', default=False)   ## Bias correction
parser.add_argument('--newa', '-na', action='store_true', help='Use new fused relu alpha', default=False)
parser.add_argument('--debug', '-d', action='store_true', help='debug mode', default=False)
args = parser.parse_args()

def setmylog(args):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    # 建立一个filehandler来把日志记录在文件里，级别为视args而定
    fh = logging.FileHandler(f'./inference/log_inference_{args.arch}.log')
    if args.debug:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    # 建立一个streamhandler来把日志打在CMD窗口上，级别为INFO以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    #将相应的handler添加在logger对象中
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info('')
    logger.info(torch.cuda.get_device_name(0))
    return logger
logger = setmylog(args)

def box_trans(bbox_list):
    # input model output format (x left_bottom ,y left_bottom , x right top,y right top)
    # output [x left top, y left top, width ,height]
    xmin,ymin,xmax,ymax=bbox_list[0],bbox_list[1],bbox_list[2],bbox_list[3]
    
    x_left_top=xmin
    y_left_top=ymin  # 图片的坐标原点在左上角，所以left top 是y min 服了!!!!
    width=xmax-xmin
    height=ymax-ymin
    return [x_left_top,y_left_top,width,height]  
# path2data="/home/ubuntu/multi/benchmark/COCO/val2017"
# path2json="/home/ubuntu/multi/benchmark/COCO/annotations/instances_val2017.json"
path2data="/home/cs433/files/benchmark/COCO/val2017"
path2json="/home/cs433/files/benchmark/COCO/annotations/instances_val2017.json"
pred2json="./inference/val_pred.json" 

class InferenceModel:
    def __init__(self, ml_logger=None):
        self.ml_logger = ml_logger
        global args, best_prec1

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if 'cuda' in args.device and torch.cuda.is_available():
            if args.seed is not None:
                torch.cuda.manual_seed_all(args.seed)
            torch.cuda.set_device(args.device_ids[0])
            cudnn.benchmark = True
        else:
            args.device_ids = None

        # create model
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == 'retinanet':
            from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
            self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
            # checkretinanet(self.model)
        elif args.arch == 'deeplabv3':
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.model = deeplabv3_resnet50(weights = 'COCO_WITH_VOC_LABELS_V1')
            # checkdeeplabv3(self.model)
        else:
            self.model = models.__dict__[args.arch](pretrained=True)
            # checkresnet(self.model)
        # set_node_names(self.model) # 给最基本部件一个internal_name，记录parent结构和名字
        logger.info(f'{self.model}')
        # exit()
        # Mark layers before relue for fusing
        if args.fused_relu:
            if 'resnet' in args.arch:
                resnet_mark_before_relu(self.model) # 给某些层创before_relu域
            if 'deeplabv3' in args.arch:
                deeplabv3_mark_before_relu(self.model)
            if 'retinanet' in args.arch:
                retinanet_mark_before_relu(self.model)

        # BatchNorm folding
        logger.info("Perform BN folding")
        search_absorbe_bn(self.model) # 把bn的absorbed设为true，并把bn整合到前面的conv2d或linear里
        QM().bn_folding = True
        
        self.model.to(args.device)
        
        if QM().enabled:
            QM().quantize_model(self.model)  # 这一步对模型参数做了quantize 和 dequantize
        cudnn.benchmark = True

        # Data loading code   加载数据
        valdir = os.path.join(args.data, 'val')  #'/home/ubuntu/multi/benchmark/ImageNet/val'

        if args.arch == 'deeplabv3':
            import utils.ext_transforms as et
            from torchvision.datasets import VOCSegmentation

            val_transform = et.ExtCompose([
                et.ExtResize(520),
                et.ExtCenterCrop(520),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
            # val_data = VOCSegmentation(root="/home/ubuntu/multi/benchmark/VOC2012/", year='2012',
            #                             image_set='val', download=False, transforms=val_transform)
            val_data = VOCSegmentation(root="/home/cs433/files/benchmark/VOC2012/", year='2012',
                                        image_set='val', download=False, transforms=val_transform)
            self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True if args.shuffle else False, num_workers=args.workers, pin_memory=True)
        elif args.arch == 'retinanet':
            self.val_loader = datasets.CocoDetection(root = path2data,annFile = path2json)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            resize = 256
            crop_size = 224
            tfs = [
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]

            self.val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose(tfs)),
                batch_size=args.batch_size, shuffle=(True if args.shuffle else False),
                num_workers=args.workers, pin_memory=True)

    def run(self):
        top1 = AverageMeter()
        top5 = AverageMeter()
        start_time = time.time()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            with tqdm(total = len(self.val_loader), leave=False) as pbar:
                for i, (input, target) in enumerate(self.val_loader):   #len=782
                    if i == 0:
                        QM().verbose = True
                    input = input.to(args.device)    #(64,3,224,224)
                    target = target.to(args.device)   #(64,)
                    # input = QM().quantize_instant(input, 'activation-1', "activation_first", verbose=False)
                    # print(max([len(torch.unique(input[:,idx,:,:])) for idx in range(input.shape[1])]))
                    # import pdb
                    # pdb.set_trace()
                    output = self.model(input)
                    QM().reset_counters()
                    QM().verbose = False

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(output, target, topk=(1, 5))
                    top1.update(float(prec1), input.size(0))
                    top5.update(float(prec5), input.size(0))
                    pbar.set_postfix({'p1': prec1})
                    pbar.update(1)

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        logger.info('time: {}'.format(time.time()-start_time))

    def rundlv3(self):
        metrics = StreamSegMetrics(21)
        start_time = time.time()
        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(self.val_loader)):
                if i == 0:
                    QM().verbose = True
                input = input.to(args.device)
                target = target.to(args.device)
                # input = QM().quantize_instant(input, 'activation-1', "activation_first", verbose=False)
                output = self.model(input)['out']

                QM().reset_counters()
                QM().verbose = False

                preds = output.detach().max(dim=1)[1].cpu().numpy()
                targets = target.cpu().numpy()

                metrics.update(targets, preds)
            score = metrics.get_results()
            logger.info(metrics.to_str(score))
            logger.info('time: {}'.format(time.time()-start_time))

    def runretina(self):
        # switch to evaluate mode
        self.model.eval()
        results = []
        start_time = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(self.val_loader)):
                if i == 0:
                    QM().verbose = True
                try:
                    image_id = target[0]['image_id']
                except:
                    continue   # logger.info(label)  有一些label是[]
                tf=transforms.ToTensor()  #这样就可以
                transformed = tf(input)   #torch.Size([3, 426, 640])  #(height*width)  
                batched = transformed.unsqueeze(0) 
                batched = batched.to(args.device)
                # batched = QM().quantize_instant(batched, 'activation-1', "activation_first", verbose=False)
                output = self.model(batched)[0]
                QM().reset_counters()
                QM().verbose = False

                # measure accuracy and record loss
                boxes=output['boxes'].cpu()
                scores=output['scores'].cpu()
                labels=output['labels'].cpu()
                threshold=0  # We will use the threshold parameter to select bounding boxes with a certain level of confidence or higher. The rule of thumb is to use 0.5
                
                num=labels.shape[0]   
                for t in range(num):    
                    score=scores[t].item()
                    if score < threshold:  # scores are sorted, so we can break  
                        break
                    box= boxes[t].tolist() 
                    box= box_trans(box)    # retinanet output bbox 和 coco dataset 的格式不一样，需要转换
                            
                    results.append({"image_id": image_id,  
                                    "category_id": labels[t].item(),
                                    "bbox":box,
                                    "score": score})
                    # fp.write(json.dumps({"image_id": image_id,  
                    #                 "category_id": labels[t].item(),
                    #                 "bbox":box,
                    #                 "score": score})+'\n')        # fp.close()
        logger.info('time: {}'.format(time.time()-start_time))
        with open(pred2json, "w") as f:
            json.dump(results, f, indent=4)
        
        cocoAnnotation = COCO(annotation_file=path2json)  #ground_truth
        cocovalPrediction = cocoAnnotation.loadRes(pred2json)
        cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")
        
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

def get_params(logger=None):
    qparams = {
        'int': {
            'clipping': args.clipping,
            'true_zero': args.preserve_zero,
            'pcq_weights': args.per_channel_quant_weights,
            'pcq_act': args.per_channel_quant_act,
            'bcorr_act': args.bias_corr_act,
            'bcorr_weight': args.bias_corr_weight,
            'vcorr_weight': args.var_corr_weight,
            'newa': args.newa,
            'logger': logger,
        }
    }  # TODO: add params for bfloat
    return qparams

if __name__ == '__main__':
    logger.info('{}'.format(args))
    with QM(args, get_params(logger)):
        im = InferenceModel(logger)  #量化了模型参数，准备了val数据
        if args.arch == 'deeplabv3':
            im.rundlv3()
        elif args.arch == 'retinanet':
            im.runretina()
        else:
            im.run()
        # QM().plotbmu()
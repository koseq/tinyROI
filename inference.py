import argparse
import os
import json
from collections import OrderedDict

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.ops as ops

from models import U2NETP
from data_loader import InferenceDataset

# from models.yolov4 import Darknet
# from models.yolov4.torch_utils import do_detect

from PyTorch_YOLOv4.utils.general import non_max_suppression,box_iou
from PyTorch_YOLOv4.models.models import *

# from non_max import filter_predictions, non_max_suppression




def findBboxes(label, original_shape, current_shape, det_size=960):
    H,W = original_shape
    _H,_W = current_shape
    contours = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    bboxes = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        xmin, ymin = (x/_W)*W, (y/_H)*H
        xmax, ymax = ((x+w)/_W)*W, ((y+h)/_H)*H
        
        bbox = [int(x) for x in [xmin,ymin,xmax,ymax]]
        bboxes.append(bbox)
    return bboxes


def isInsidePoint(bbox, point):
    xmin, ymin, xmax, ymax = bbox
    x, y = point
    if xmin<=x<=xmax and ymin<=y<=ymax:
        return True
    
def isInsideBbox(inner_bbox, outer_bbox):
    xmin,ymin,xmax,ymax = inner_bbox
    p1, p2, p3, p4 = (xmin,ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)
    return all([isInsidePoint(outer_bbox, point) for point in [p1,p2,p3,p4]])
    

def getDetectionBbox(bbox,  max_H, max_W, det_size=960):
    xmin, ymin, xmax, ymax = bbox
    xc = (xmax+xmin)//2
    yc = (ymin+ymax)//2

    xmin = max(xc - det_size//2, 0)
    ymin = max(yc - det_size//2, 0)

    xmax = min(xmin+det_size, max_W)
    ymax = min(ymin+det_size, max_H)
    return [xmin,ymin,xmax,ymax]

                
        
def getDetectionBboxesAll(bboxes, max_H, max_W, det_size=960):
    det_bboxes = []
    for bbox in bboxes:
        det_bboxes.append(getDetectionBbox(bbox,  max_H, max_W, det_size=det_size))
    return det_bboxes
    

def getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=960):
    det_bboxes = []
    for bbox in bboxes:
        if any([isInsideBbox(bbox, det_bbox) for det_bbox in det_bboxes]):
            continue
            
        det_bboxes.append(getDetectionBbox(bbox,  max_H, max_W, det_size=det_size))
    return det_bboxes


def getDetectionBboxesSorted(bboxes, max_H, max_W, det_size=960):
    _det_bboxes = [getDetectionBbox(bbox, max_H, max_W, det_size=det_size) for bbox in bboxes]
    hits = {i: 0 for i in range(len(_det_bboxes))}
    for i, det_bbox in enumerate(_det_bboxes):
        hits[i]+=sum([isInsideBbox(bbox, det_bbox) for bbox in bboxes])
    # print(hits)
    if all([x==1 for x in hits.values()]):
        return _det_bboxes
    elif any([x==len(bboxes) for x in hits.values()]):
        fnd = list(hits.keys())[list(hits.values()).index(len(bboxes))]
        # print(fnd)
        return [_det_bboxes[fnd]]
    else:
        hits = dict(sorted(hits.items(), key=lambda item: item[1], reverse=True))
        bboxes = [bboxes[i] for i in hits.keys()]
        return getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=det_size)
        

        
        
def getDetectionBboxes(bboxes, max_H, max_W, det_size=960, bbox_type='naive'):
    if bbox_type == 'naive':
        return getDetectionBboxesNaive(bboxes, max_H, max_W, det_size=det_size)
    elif bbox_type == 'all':
        return getDetectionBboxesAll(bboxes, max_H, max_W, det_size=det_size)
    elif bbox_type == 'sorted':
        return getDetectionBboxesSorted(bboxes, max_H, max_W, det_size=det_size)
    else:
        raise NotImplementedError
        
        
        
def getSlidingWindowBBoxes(max_H, max_W, overlap_frac, det_size=960):
    n_w = math.ceil(W_orig / (det_size*(1-overlap_frac)))
    n_h = math.ceil(H_orig / (det_size*(1-overlap_frac)))

    XS = [det_size*(1-overlap_frac)*n for n in range(n_w)]
    YS = [det_size*(1-overlap_frac)*n for n in range(n_h)]

    bboxes = []
    for xmin in XS:
        for ymin in YS:
            xmin,ymin = [int(_) for _ in [xmin,ymin]]
            xmax,ymax = min(xmin+det_size, max_W), min(ymin+det_size, max_H)
            bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes


def NMS(prediction, iou_thres,  redundant=True, merge=False, max_det=300, agnostic=False):
    # https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/general.py
    # output = [torch.zeros(0, 6)] * len(prediction)
    x = prediction.clone()
        
    # Batched NMS
    boxes, scores, c = x[:, :4], x[:, 4], x[:, 5] * (0 if agnostic else 1)

    i = ops.batched_nms(boxes, scores, c, iou_thres)
    # i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    if merge and (1 < x.shape[0] < 3E3):  # Merge NMS (boxes merged using weighted mean)
        # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        # iou = bbox_iou(torch.unsqueeze(boxes[i], 0), boxes,DIoU=True) > iou_thres
        iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        weights = iou * scores[None]  # box weights
        x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        if redundant:
            i = i[iou.sum(1) > 1]  # require redundancy

        # output[xi] = x[i]
    return x[i] #output
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ROI
    parser.add_argument('--roi_ckpt', type=str, default='trained_models/u2netp.pth')
    parser.add_argument('--roi_inf_size', type=int, default=576)
    parser.add_argument('--roi_th', type=float, default=0.5)
    parser.add_argument('--dilate', default=False, action='store_true')
    parser.add_argument('--k_size', type=int, default=7)
    parser.add_argument('--iter', type=int, default=2)
    parser.add_argument('--bbox_type', type=str, default='naive', choices=['all', 'naive', 'sorted'])
    
    # net_det
    parser.add_argument('--det_ckpt', type=str, default='trained_models/mtsd001_best.weights')
    parser.add_argument('--det_cfg', type=str, default='trained_models/mtsd001-960.cfg')
    parser.add_argument('--det_inf_size', type=int, default=960)
    
    # NMS
    parser.add_argument('--conf_thresh', type=float, default=0.005)
    parser.add_argument('--iou_thresh', type=float, default=0.45)
    parser.add_argument('--second_nms', default=False, action='store_true')
    
    # sliding window
    parser.add_argument('--overlap_frac', type=float, default=0.05)

    
    # general
    parser.add_argument('--input_files', type=str, default='input_list.txt')
    parser.add_argument('--mode', type=str, default='roi', choices=['det', 'roi', 'sw'])
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--save_results', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default='detections')
    parser.add_argument('--out_json', type=str, default='labels.json')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 and not args.cpu else 'cpu')
    
    
    if args.save_results:
        os.makedirs(args.out_dir, exist_ok=True)

    
    flist = sorted([x.rstrip() for x in open(args.input_files) if os.path.isfile(x.rstrip())])
    dataset = InferenceDataset(
        img_name_list=flist,
        roi_inf_size=args.roi_inf_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    if args.mode == 'roi':
        net_roi = U2NETP(3,1)
        net_roi.load_state_dict(torch.load(args.roi_ckpt, map_location='cpu'))
        net_roi.to(device)
        net_roi.eval()
    
    
    net_det = Darknet(args.det_cfg, img_size=(args.det_inf_size, args.det_inf_size)).to(device)
    load_darknet_weights(net_det, args.det_ckpt)
    net_det.eval()
    
    num_dets = 0
    object_id = 0
    annotations = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            intensor_roi = data['image_roi'].float().to(device)
            intensor_det = data['image_det'].float().to(device)
            H_orig, W_orig = data['height'].item(), data['width'].item()
            original_shape = (H_orig, W_orig)

            if args.save_results:
                image = cv2.imread(flist[i])
                
                
            if args.mode == 'det':
                tr = T.Resize((args.det_inf_size, args.det_inf_size))
                intensor_detection = tr(intensor_det)
                inf_out, train_out = net_det(intensor_detection, augment=False)
                num_dets+=1

                img_output = non_max_suppression(
                    inf_out, 
                    conf_thres = args.conf_thresh, 
                    iou_thres = args.iou_thresh,
                    merge = True,
                    agnostic = False)[0] # batch==1
                img_output = img_output.to(device)
                img_output[:,:4] *= torch.tensor([W_orig/args.det_inf_size,H_orig/args.det_inf_size,W_orig/args.det_inf_size,H_orig/args.det_inf_size]).to(device)
                
                
            elif args.mode == 'roi':
                d0 = net_roi(intensor_roi)[0]

                d0 = d0[0,0,...]
                d0 = torch.where(d0>args.roi_th, 1.0, 0.0)
                d0 = 255*d0.detach().cpu().numpy().astype(np.uint8)

                if args.dilate:
                    kernel = np.ones((args.k_size,args.k_size), np.uint8)
                    d0 = cv2.dilate(d0,kernel,iterations = args.iter)

                bboxes = findBboxes(d0, original_shape, (args.roi_inf_size, args.roi_inf_size))
                bboxes_det = getDetectionBboxes(bboxes, H_orig, W_orig, det_size=args.det_inf_size, bbox_type=args.bbox_type)
            
            
            elif args.mode == 'sw':
                bboxes_det = getSlidingWindowBBoxes(H_orig, W_orig, args.overlap_frac, det_size=args.det_inf_size)
                
            if args.mode == 'sw' or args.mode =='roi':
                img_output = None
                for bbox_det in bboxes_det:
                    xmin, ymin, xmax, ymax = bbox_det
                    intensor_detection = torch.zeros((1, 3, args.det_inf_size, args.det_inf_size)).to(device)
                    intensor_detection[...,:ymax-ymin,:xmax-xmin] = intensor_det[...,ymin:ymax,xmin:xmax]
                    H,W = intensor_detection.shape[2:]
                    
                    inf_out, train_out = net_det(intensor_detection, augment=False)
                    num_dets+=1
                    output = non_max_suppression(
                        inf_out, 
                        conf_thres = args.conf_thresh, 
                        iou_thres = args.iou_thresh,
                        merge = True,
                        agnostic = False)[0] # batch==1

                    output[:,:4] = output[:,:4].to(device) + torch.tensor([xmin,ymin,xmin,ymin]).to(device)
                    if img_output is None:
                        img_output = output.clone()
                    else:
                        img_output = torch.cat((img_output.to(device), output.to(device)), 0)
                        
                if args.second_nms and img_output is not None:
                    img_output = NMS(img_output, iou_thres=args.iou_thresh, redundant=args.redundant, merge=args.merge, max_det=args.max_det, agnostic=args.agnostic) 
                        
            
            if img_output is not None:
                for det in img_output:
                    det = det.cpu().numpy()
                    _xmin,_ymin,_xmax,_ymax = det[:4]
                    w,h = _xmax-_xmin, _ymax-_ymin
                    area = w*h

                    dct = {
                        "id": int(object_id),
                        "image_id": i+1,
                        "category_id": int(det[-1]),
                        "bbox" : [float(_) for _ in [_xmin, _ymin, w, h]],
                        "score": float(det[-2]),
                        "area": float(area)
                    }

                    annotations.append(dct)
                    object_id+=1
                    if args.save_results:
                        image = cv2.rectangle(image, (int(_xmin),int(_ymin)), (int(_xmax),int(_ymax)), (0,0,255), 4)
                    
            if args.save_results:
                cv2.imwrite('%s/%03d.png' % (args.out_dir, i), image)  
            
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
        
    print('Number of yolov4 detections (%s): %d' % (os.path.basename(args.out_json), num_dets))
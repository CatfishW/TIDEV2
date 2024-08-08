# -*- coding: utf-8 -*-
"""
TIDE: Few-Shot
"""
import os

import numpy as np
import cv2
import torch 
import supervision as sv
from PIL import Image
from model.tide import TIDE
from model.encoder import HybridEncoder
from model.decoder import TideTransformer
from backbone.resnet import ResNet
from config import args
import os
import torch
from misc.utils import load_weight
from model.utils import nested_tensor_from_tensor_list
from config import args
import dataset.transforms as T
def NMS(boxes, scores,cls_ids,scores_pred,iou_threshold):
    """Non-maximum suppression.
    ==============================================================================================
    args:
        boxes: list of list,coordinates of the bounding boxes
        scores: list of float,confidence scores
        cls_ids: list of int,class ids
        scores_pred: list of float,confidence scores
        iou_threshold: float,threshold for the nms iou

    ==============================================================================================
    return:
        list of list,coordinates of the bounding boxes
        cls_ids: list of int,class ids
    ==============================================================================================
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    cls_ids = np.array(cls_ids)
    scores_pred = np.array(scores_pred)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    

    while order.size > 0:

        i = order[0]
    

        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes[keep],cls_ids[keep],scores_pred[keep]


class TIDEAPI:
    """TIDE API class.
    ==============================================================================================
    including the following functions:
        __init__:initialize the model
        add_support:add support images to the model
        predict:predict the bounding boxes of the query image
    ==============================================================================================
    in predict function return the following information:
        dict containing:
            'status':True if the prediction is successful,otherwise false
            'result':a list containing cls_id,probability and x1,y1,x2,y2 coordinates of the bounding box
    ==============================================================================================
    """
    def __init__(self, param):
        """Initialize TIDE.
        ==============================================================================================
        args:
            param['model_path']: str,path to the model
            param['theta']: float,threshold for the confidence score
            param['iou']: float,threshold for the nms iou
        ==============================================================================================
        """
        self.output_theta = param['theta']
        self.output_iou = param.get('iou',None)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.support_feature_extractor = args.support_backbone.to(self.device)
        self.model = TIDE(
                backbone=args.backbone,
                encoder=HybridEncoder(in_channels=args.backbone_dims,
                                    BMHA=args.BMHA,
                                    dim_feedforward=args.encoder_dim_feedforward,
                                    num_fusion_layers=args.num_fusion_layers,
                                    ),
                decoder=TideTransformer(num_classes=args.max_support_len,
                                        feat_channels=[256, 256, 256],
                                        num_denoising=args.num_denoising,
                                        normalize=args.query_support_norm,
                                        dim_feedforward=args.decoder_dim_feedforward,#args.dim_feedforward,
                                        ),
                                        multi_scale=None).to(args.device)
        self.model.eval()
        self.model = load_weight(args,self.model)
        self.support_list = [torch.ones(args.support_feat_dim).to(self.device) / torch.linalg.norm(torch.ones(args.support_feat_dim)).to(self.device)]
        self.support_set = {}
        self.support_images = None
        self.class_id = None
        self.mapping = [0]
        self.id2label = param['id2label']
        self.support_samples = torch.stack(self.support_list)

    def add_support(self,param):
        """Add support images to the model.
        ==============================================================================================
        args:
            param['support_images']: list of numpy.ndarray
            param['class_id']:dict of class id
            param[support_status]:bool,whether the support images are added successfully
        ==============================================================================================
        return:
            True/False:whether the support images are added successfully.
        ==============================================================================================
        """
        try:
            self.support_images = param['support_images']
            self.class_id = param['class_id']
            
            for cls,support_img_CV2 in zip(self.class_id[1:],param['support_images']):
                if cls not in self.support_set:
                    self.support_set[cls] = []
                    self.mapping.append(cls)
                hs,ws,c = support_img_CV2.shape
                if (hs > 640 or ws > 640 ) and param.get('square_resize',True):
                    temp = self.support_feature_extractor(load_image(square_resize(support_img_CV2,(224,224))).unsqueeze(0).to(self.device))[0]
                else:
                    temp = self.support_feature_extractor(load_image(resize_to_closest_14x(support_img_CV2)).unsqueeze(0).to(self.device))[0]
                temp = temp/torch.linalg.norm(temp)
                self.support_set[cls].append(temp)
                if len(self.support_set[cls]) == 0:
                    self.support_set.pop(cls)
            self.support_list = [torch.ones(args.support_feat_dim).to(self.device) / torch.linalg.norm(torch.ones(args.support_feat_dim)).to(self.device)] if len(self.support_list)>1 else self.support_list
            for cls in self.support_set:
                shots_list = []
                for feat in self.support_set[cls]:
                    shots_list.append(feat)
                avg_shots = torch.mean(torch.stack(shots_list),dim=0) if len(shots_list)>1 else shots_list[0]
                self.support_list.append(avg_shots)
            self.support_samples = torch.stack(self.support_list)
            self.support_samples = nested_tensor_from_tensor_list(self.support_samples.unsqueeze(0).transpose(1,2)).to(device=self.device)
            param['support_status'] = True
            return True
        except Exception as e:
            print(e)
            param['support_status'] = False
            return False

    def del_support(self,param):
        if param['del_support']:
            self.support_set = {}
            self.mapping = [0]

    def predict(self,param):
        """predict the bounding boxes of the query image.
        ==============================================================================================
        args:
            param['query_image']: numpy.ndarray
        ==============================================================================================
        return:
            dict containing:
            'status':True if the prediction is successful,otherwise false
            'result':a list containing cls_id,probability and x1,y1,x2,y2 coordinates of the bounding box
        ==============================================================================================
        """
        if not param['support_status']:
            return {'status':False,'result':None}
        try:
            
            query_img_np = param['query_image']
            h, w, c = query_img_np.shape
            query_img_np_org = np.copy(query_img_np)
            query_img_np = cv2.resize(query_img_np, (640, 640))
            
            query_img = load_image(query_img_np)
            query_sample = nested_tensor_from_tensor_list([query_img]).to(device=self.device)
            outputs = self.model(query_sample, self.support_samples)
            # post-process
            logits = torch.nn.functional.softmax(outputs["pred_logits"][0]).cpu()
            boxes_pred = outputs["pred_boxes"][0].cpu()
            del outputs
            cls_ids = []
            boxes_pred_sv = []
            scores_pred = []
            for logit, box in zip(logits, boxes_pred):
                xywh = [x.item() for x in box.cpu()]
                box = xywh_2_xyxy(xywh, w, h)
                prompt_cls = logit.argmax().item()
                score = logit[prompt_cls].item()
                if prompt_cls > 0 and score > self.output_theta:
                    cls_ids.append(prompt_cls)
                    boxes_pred_sv.append(box)
                    scores_pred.append(score)
                    print('class:', prompt_cls, 'score:', score)
            if boxes_pred_sv:
                box_sv = np.array(boxes_pred_sv)
                box_further_nms,cls_ids_nms,scores_pred_nms = NMS(boxes_pred_sv, scores_pred, cls_ids,scores_pred,self.output_iou)
                res_with_nms = sv.Detections(box_further_nms, class_id=cls_ids_nms, confidence=scores_pred_nms)
                final_res = []
                for label_id,prob,(x1,y1,x2,y2) in zip(res_with_nms.class_id,res_with_nms.confidence,res_with_nms.xyxy):
                    final_res.append([self.mapping[label_id],prob,x1,y1,x2,y2])
                #if windows platform
                if os.name == 'nt':
                    box_annotator = sv.BoxAnnotator()
                    box_annotator.annotate(query_img_np_org, res_with_nms, labels=[str(self.mapping[x]) for x in res_with_nms.class_id])
                    cv2.imshow('test', query_img_np_org)
                    cv2.waitKey(0)
                return {'status':True,'result':final_res}
            else:
                return {'status':True,'result':None}
        except Exception as e:
            print(e)
            return {'status':False,'result':None,'error':e}
def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)
    return [x0, y0, x1, y1]
def resize_to_closest_14x(cv2img):
    h, w = cv2img.shape[:2]
    h_new = int(np.ceil(h / 14) * 14)
    w_new = int(np.ceil(w / 14) * 14)
    cv2img = cv2.resize(cv2img, (w_new, h_new))
    return cv2img
def square_resize(img, dsize,value=(0,0,0)):
    """
    按照图片的长边扩充为方形, 再resize到指定大小
    """
    h, w, _ = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    if h > w:
        diff = h - w
        left = int(diff / 2)
        right = diff - left
    else:
        diff = w - h
        top = int(diff / 2)
        bottom = diff - top
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    return cv2.resize(img_new, dsize)
@torch.no_grad()
def load_image(image):
    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ]
                    )
    image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_transformed,_= transform(image_pillow,None)
    return image_transformed
if __name__ == '__main__':
    param = {
            'theta': 0.0,
            'iou': 0.3,
            'query_image': None,
            'support_images': None,
            'class_id': [0,1,1,1,2,5],
            'support_status': False,
            'id2label': None,#{0:'background',1:'zebra',2:'giraffe'},
            'square_resize':True,
            }
    tide= TIDEAPI(param)
    param['query_image'] = cv2.imread('./dataset/query_set/1.jpg')
    param['support_images'] = [
        cv2.imread('./dataset/support_set/1/cxk.png'),
        cv2.imread('./dataset/support_set/1/wj.png'),
        cv2.imread('./dataset/support_set/1/zjl.png'),
        cv2.imread('./dataset/support_set/2/racket.png'),
    ]
    tide.add_support(param)
    res = tide.predict(param)
    print(res)


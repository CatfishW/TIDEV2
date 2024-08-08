import torch
import sys
import json
import cv2
import os
from pandas import DataFrame as df
import pandas as pd
import numpy as np
import pycocotools
import time
import glob
def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)
    return [x0, y0, x1, y1]
coco_format = {}
path_images = '/root/workspace/dataset/project_testdata_all/images'
path_labels = '/root/workspace/dataset/project_testdata_all/labels'
txts = glob.glob(path_labels+'/*.txt')
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [],
    "licenses": [],
}
ann_id = 0
cat_set = []
for txt in txts:
    img_id = int(txt[-23:-4])
    img_path = os.path.join(path_images, txt[-23:-4] + '.jpg')
    img = cv2.imread(img_path)
    h, w, c = img.shape
    coco_data['images'].append({
        "id": img_id,
        "width": w,
        "height": h,
        "file_name": img_path,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": ""
    })
    
    with open(txt,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')
            cls = line[0]
            if cls not in cat_set:
                coco_data['categories'].append({
                    "id": int(cls),
                    "name": cls,
                    "supercategory": ""
                })
                cat_set.append(cls)
            box = [float(x) for x in line[1:5]]

            box = xywh_2_xyxy(box,w,h)
            #xyxy to xywh
            box[2] = box[2]-box[0]
            box[3] = box[3]-box[1]
            # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),5)
            # img = cv2.resize(img,(640,480))
            # cv2.imshow('test',img)
            # cv2.waitKey(5)
            coco_data['annotations'].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": box,
                "area": box[2]*box[3],
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id+=1
    #progress bar
    print('progress:',len(coco_data['images']),'/',len(txts))
with open('./coco2017/annotations/instances_project_all_coco.json','w') as f:
    json.dump(coco_data,f)
print('done')
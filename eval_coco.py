import torch
import torch.nn as nn
from typing import Iterable, List, Optional
import torch.nn.functional as F
import json
import tqdm
import numpy as np
import supervision as sv
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pickle
import pandas as pd
import cv2
import os
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    test_ann_path: str,
    test_image_folder_path: str,
    batch_size: int = 1,
):
    model.eval()
    # for querys, supports, querys_wh_tensor,supports_wh_tensor,targets in metric_logger.log_every(data_loader, print_freq, header):
    print('PROGRESS:')
    data_loader = tqdm.tqdm(data_loader)
    predictions = []
    instances = json.load(open(test_ann_path, 'r'))
    gt_instances = instances.copy()
    gt_instances['annotations'] = []
    for index, (sample_query, sample_support, targets, hw, target_to_gt_mapping, imgid, ann) in enumerate(data_loader):
        # print('init mem:',torch.cuda.max_memory_allocated() / MB)
        sample_query = sample_query.to(device)
        # print('mem:',torch.cuda.max_memory_allocated() / MB)
        sample_support = sample_support.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(sample_query, sample_support, targets)
        #change the name of each ann: catid->catgory_id, imgid->image_id
        prediction = postprocess(output, 
                    target_to_gt_mapping, 
                    imgid, 
                    ann, 
                    gt_instances,
                    batch_size=batch_size,
                    test_image_folder_path=test_image_folder_path)
        predictions.extend(prediction)
    with open('gt_instances.json', 'w') as f:
        json.dump(gt_instances, f)
    return predictions
                    

def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)
    return [x0, y0, x1, y1]

@torch.no_grad()
def postprocess(outputs,mapping, imgids, ann,gt_instances,batch_size=1,test_image_folder_path=None):
    predictions = []
    #print(len(gt_instances['annotations']))
    cates_pos = []
    for index in range(batch_size):
        try:
            logits = torch.nn.functional.softmax(outputs["pred_logits"][index])
        except:
            break
        boxes = outputs["pred_boxes"][index]
        
        query_has_bbox = False
        cls_ids = []
        boxes_pred_sv = []
        scores_pred = []
        if 'voc' in test_image_folder_path.lower():
            query_img_cv2 = cv2.imread(os.path.join(
                test_image_folder_path, str(ann[index][0]['image_id'])[:4]+'_'+str(ann[index][0]['image_id'])[4:]+'.jpg'))
        else:
            query_img_cv2 = cv2.imread(os.path.join(
                    test_image_folder_path, str(ann[index][0]['image_id']).zfill(12)+'.jpg'))   
        h,w,c = query_img_cv2.shape
        for logit, box in zip(logits, boxes):
            xywh = [x.item() for x in box.cpu()]
            xyxy = xywh_2_xyxy(xywh, w, h)
            prompt_idx = logit.argmax().item()
            score = logit[prompt_idx].item()
            if score > 0.0 and prompt_idx > 0:
                cls_ids.append(prompt_idx)
                boxes_pred_sv.append(xyxy)
                scores_pred.append(score)
                query_has_bbox = True
        for i in range(len(ann[index])):
            gt_instances['annotations'].append(ann[index][i])
            cates_pos.append(str(ann[index][i]['category_id']))
        if query_has_bbox:
            box_sv = np.array(boxes_pred_sv)
            res_with_nms = sv.Detections(box_sv, class_id=np.array(
                cls_ids), confidence=np.array(scores_pred)).with_nms(threshold=0.3)
            try:
                annotator = sv.BoxAnnotator()
                mapped_cls_id = [str(mapping[index].get(x)) for x in res_with_nms.class_id]
                annotator.annotate(query_img_cv2, res_with_nms, labels=mapped_cls_id)
            except Exception as e:
                print(e)
                pass
            #cv2.imwrite(f'./test_{imgids[index]}.jpg', query_img_cv2)
            # cv2.imwrite(f'eval_output/project_gt_{imgids[index]}.jpg', query_img_gt)
            #cv2.imwrite(f'eval_output/project_org_{imgids[index]}_sv.jpg', query_img_org)
            
            for box, cls_id, score in zip(res_with_nms.xyxy, res_with_nms.class_id, res_with_nms.confidence):
                prediction = {}
                prediction['image_id'] = imgids[index]
                x = box[0]
                y = box[1]
                width = box[2] - box[0]
                height = box[3] - box[1]
                xywh = [x, y, width, height]
                prediction['bbox'] = xywh
                prediction['score'] = score
                prediction['category_id'] = mapping[index].get(cls_id)
                # if prediction['category_id'] is None or prediction['category_id'] not in cates_pos:
                #     continue
                predictions.append(prediction)
    return predictions


def get_eval_res(model,shot,test_save_path,test_ann_path,test_image_folder_path,test_batch_size,data_loader_test,device,epoch=None):
    print(f"Start {shot} shot Evaluating, results will be saved to {shot}{test_save_path}...")
    predictions = evaluate(model, data_loader_test,device,test_ann_path,test_image_folder_path,test_batch_size)
    coco = COCO('./gt_instances.json')
    cocoDt = coco.loadRes(predictions)
    # 创建COCO评估器对象a
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    #save to pkl
    with open('cocoEval.pkl', 'wb') as f:
        pickle.dump(cocoEval, f)
    #cocoEval.params.iouThrs = [0.3]
    # 运行评估
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if not os.path.exists('eval_csv'):
        os.makedirs('eval_csv')
    result = pd.DataFrame(cocoEval.stats,index=['AP@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                'AP@[ IoU=0.50      | area=   all | maxDets=100 ]',
                                                'AP@[ IoU=0.75      | area=   all | maxDets=100 ]',
                                                'AP@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                'AP@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                'AP@[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                                                'AR@[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                                                'AR@[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                                                'AR@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                'AR@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                'AR@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                'AR@[ IoU=0.50:0.95 | area= large | maxDets=100 ]'])
    result.to_csv('./eval_csv/'+f'EPOCH-{epoch}-'+str(shot)+test_save_path)
    #specific class
    if 'voc' not in test_image_folder_path.lower() and 'project' not in test_image_folder_path.lower() : 
        cocoEval.params.catIds = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    elif 'project' in test_image_folder_path.lower():
        #AP30
        cocoEval.params.iouThrs = [0.3]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    if 'project' in test_image_folder_path.lower():
        result = pd.DataFrame(cocoEval.stats,index=['AP@[ IoU=0.3 | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.3      | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.3      | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.3 | area= small | maxDets=100 ]',
                                                    'AP@[ IoU=0.3 | area=medium | maxDets=100 ]',
                                                    'AP@[ IoU=0.3 | area= large | maxDets=100 ]',
                                                    'AR@[ IoU=0.3 | area=   all | maxDets=  1 ]',
                                                    'AR@[ IoU=0.3 | area=   all | maxDets= 10 ]',
                                                    'AR@[ IoU=0.3 | area=   all | maxDets=100 ]',
                                                    'AR@[ IoU=0.3 | area= small | maxDets=100 ]',
                                                    'AR@[ IoU=0.3 | area=medium | maxDets=100 ]',
                                                    'AR@[ IoU=0.3 | area= large | maxDets=100 ]'])
        result.to_csv('./eval_csv/'+f'EPOCH-{epoch}-'+'AP30-'+str(shot)+test_save_path)
        AP30_best_res_value = cocoEval.stats[0]
        print('best AP30:',AP30_best_res_value)
        #per class AP
        for idx,cls in enumerate(coco.getCatIds()):
            cocoEval.params.catIds = [cls]
            cocoEval.params.iouThrs = [0.3]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            res = cocoEval.stats.tolist()
            res = np.array(res)*100
            res = np.insert(res,0,cls,axis=0)
            res = res.reshape(1,13)
            if idx == 0:
                tab = pd.DataFrame(res,
                                columns=['cls', 'AP@[IoU=0.3:0.3]', 'AP@[IoU=0.50]', 'AP@[IoU=0.75]', 'AP@[S]',
                                        'AP@[M]', 'AP@[L]', 'AR@[IoU=0.3:0.3]', 'AR@[IoU=0.3:0.3]',
                                    'AR@[IoU=0.3:0.3]', 'AR@[S]', 'AR@[M]', 'AR@[L]'])
            else:
                temp = pd.DataFrame(res,
                                columns=['cls', 'AP@[IoU=0.3:0.3]', 'AP@[IoU=0.50]', 'AP@[IoU=0.75]', 'AP@[S]',
                                        'AP@[M]', 'AP@[L]', 'AR@[IoU=0.3:0.3]', 'AR@[IoU=0.3:0.3]',
                                    'AR@[IoU=0.3:0.3]', 'AR@[S]', 'AR@[M]', 'AR@[L]'])
                tab = pd.concat([tab,temp])
        
        tab.to_csv('./eval_csv/'+f'EPOCH-{epoch}-'+'AP30-perclass-'+str(shot)+test_save_path)
    else:
        result = pd.DataFrame(cocoEval.stats,index=['AP@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.50      | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.75      | area=   all | maxDets=100 ]',
                                                    'AP@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                    'AP@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                    'AP@[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                                                    'AR@[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                                                    'AR@[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                                                    'AR@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                                                    'AR@[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                                                    'AR@[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                                                    'AR@[ IoU=0.50:0.95 | area= large | maxDets=100 ]'])
        result.to_csv('./eval_csv/'+f'EPOCH-{epoch}-'+'20cls-'+str(shot)+test_save_path)
    print(f"Finished {shot} Evaluating, results saved to {test_save_path} and 20cls-{test_save_path}...")
    return AP30_best_res_value
    

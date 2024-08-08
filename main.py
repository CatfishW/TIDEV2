from model.tide import TIDE
from model.encoder import HybridEncoder,Tidev1Encoder
from model.decoder import TideTransformer,Tidev1Decoder
from backbone.resnet import ResNet
from model.criterion import SetCriterion
from model.matcher import HungarianMatcher
from model.segmentation import DETRsegm
from config import args
from dataset.Dataset import COCODataset,collate_fn
from dataset.Dataset_avg import collate_fn_avg
from engine import train_one_epoch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
from eval_coco import evaluate,get_eval_res
import torch
import time
import os
import pickle
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import DataLoader
from misc import dist
from misc.utils import load_weight
#for rank=0
if args.gpu_num == 1:device_ids =[0]
elif args.gpu_num == 2:device_ids =[0,1]
elif args.gpu_num == 3:device_ids =[0,1,2]
elif args.gpu_num == 4:device_ids =[0,1,2,3]
else:raise NotImplementedError
torch.multiprocessing.set_sharing_strategy('file_system')
def train():
    if args.local_rank==0:
        model = TIDE(   
                backbone=args.backbone,
                encoder=HybridEncoder(in_channels=args.backbone_dims,
                                    BMHA=args.BMHA,
                                    raw_support_feat_dim=args.support_feat_dim,
                                    dim_feedforward=args.encoder_dim_feedforward,
                                    num_fusion_layers=args.num_fusion_layers,
                                    use_mask_head=args.use_mask_head,
                                    use_text = args.use_text,
                                    )if args.version == 'v2' else Tidev1Encoder(num_layers=6,num_queries=args.num_queries,fusion=args.fusion,num_feature_level=args.num_feature_level),                                                                            
                decoder=TideTransformer(num_classes=args.max_support_len,
                                        raw_support_feat_dim=args.support_feat_dim,
                                        feat_channels=[256, 256, 256],
                                        num_denoising=args.num_denoising,
                                        align_loss=args.use_align_loss,
                                        normalize=args.query_support_norm,
                                        num_queries=args.num_queries,
                                        mask_head=args.use_mask_head,
                                        dim_feedforward=args.decoder_dim_feedforward,
                                        )if args.version=='v2' else Tidev1Decoder(num_layers=6,num_queries=args.num_queries,num_feature_levels=args.num_feature_level),                                                              
                multi_scale=None,#[490, 518, 546, 588, 616, 644],
                l2_norm=args.query_support_norm,
                version=args.version,
                backbone_num_channels=args.backbone_dims,
                freeze_encoder=args.freeze_encoder,
                num_feature_levels=args.num_feature_level
                )
    else:
        model = TIDE(
                backbone=args.backbone,
                encoder=HybridEncoder(in_channels=args.backbone_dims,
                                      raw_support_feat_dim=args.support_feat_dim,
                                    BMHA=args.BMHA,
                                    dim_feedforward=args.encoder_dim_feedforward,
                                    num_fusion_layers=args.num_fusion_layers,
                                    use_mask_head=args.use_mask_head,
                                    use_text = args.use_text,
                                    )if args.version == 'v2' else Tidev1Encoder(num_layers=6,num_queries=args.num_queries,fusion=args.fusion,num_feature_level=args.num_feature_level),
                decoder=TideTransformer(num_classes=args.max_support_len,
                                        raw_support_feat_dim=args.support_feat_dim,
                                        feat_channels=[256, 256, 256],
                                        num_denoising=args.num_denoising,
                                        align_loss=args.use_align_loss,
                                        normalize=args.query_support_norm,
                                        num_queries=args.num_queries,
                                        mask_head=args.use_mask_head,
                                        dim_feedforward=args.decoder_dim_feedforward
                                        )if args.version=='v2' else Tidev1Decoder(num_layers=6,num_queries=args.num_queries,num_feature_levels=args.num_feature_level),
                                                                               
                multi_scale=None,#[490, 518, 546, 588, 616, 644],
                l2_norm=args.query_support_norm,
                version=args.version,
                backbone_num_channels=args.backbone_dims,
                freeze_encoder=args.freeze_encoder,
                num_feature_levels=args.num_feature_level
                ).to(args.device)
    model.train()
    matcher = HungarianMatcher(args.weight_dict,
                               use_focal_loss=args.use_focal_loss,
                               )
    if args.use_mask_head:
        model = DETRsegm(model, freeze_tide=(args.frozen_weights is not None)).to(args.device)
        criterion = SetCriterion(num_classes=args.max_support_len, 
                                matcher=matcher, 
                                weight_dict=args.weight_dict, 
                                eos_coef=args.eos_coef, 
                                losses=args.losses,
                                )
    else:
        criterion = SetCriterion(num_classes=args.max_support_len, 
                                matcher=matcher, 
                                weight_dict=args.weight_dict, 
                                eos_coef=args.eos_coef, 
                                losses=args.losses,
                                )
    if args.local_rank>0:
        model = dist.warp_model(model,find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=device_ids)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_drop_gamma)
    if ',' in args.test_dino_feats_folder_path:
            test_dino_feats_folder_paths = args.test_dino_feats_folder_path.split(',')
    else:
            test_dino_feats_folder_paths = [args.test_dino_feats_folder_path]
    for shot,path in zip(list(args.shots.split(',')),test_dino_feats_folder_paths):
        print("============loading evaluation dataset============")
        dataset_test = args.coco_test_dataset(args.test_ann_path, 
                                        args.test_image_folder_path, 
                                        path,
                                        aug=args.aug, 
                                        strong_aug=False,
                                        extra_shots=args.extra_shots,
                                        support_norm=args.support_norm,
                                        train=False,
                                        return_mask=args.use_mask_head,
                                        #mixed_support_selection = args.use_mixed_support_selection
                                        )
    data_loader_test = DataLoader(
                dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=collate_fn_avg)
    if args.eval_only:
        for shot,path in zip(list(args.shots.split(',')),test_dino_feats_folder_paths):
            dataset_test = args.coco_test_dataset(args.test_ann_path, 
                                            args.test_image_folder_path, 
                                            path,
                                            aug=args.aug, 
                                            strong_aug=False,
                                            extra_shots=args.extra_shots,
                                            support_norm=args.support_norm,
                                            train=False,
                                            return_mask=args.use_mask_head,
                                            #mixed_support_selection = args.use_mixed_support_selection
                                            )
            data_loader_test = DataLoader(
            dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=collate_fn_avg)
            if args.resume is not None:
                model_without_ddp = load_weight(args,model_without_ddp)
            if dist.is_main_process():
                res = get_eval_res(model,shot,
                            args.test_save_path,
                            args.test_ann_path,
                            args.test_image_folder_path,
                            args.test_batch_size,
                            data_loader_test,
                            args.device,
                            epoch=-1)
                if res:
                    print("best AP30:",res)
        exit()
    print("============loading train dataset============")
    if args.dataset_name == 'coco':
        dataset_train = args.coco_dataset(  args.coco_ann_path, 
                                                args.coco_image_folder_path, 
                                                args.coco_dino_feats_folder_path, 
                                                aug=args.aug, 
                                                strong_aug=args.strong_aug,
                                                extra_shots=args.extra_shots,
                                                support_norm=args.support_norm,
                                                train=args.training,
                                                return_mask=args.use_mask_head,
                                                mixed_support_selection = args.use_mixed_support_selection,
                                                support_feat_dim=args.support_feat_dim,
                                                val_cls_ids=args.val_cls_ids,
                                                region_detect=args.region_detect,
                                                use_text = args.use_text,
                                                )
    elif args.dataset_name == 'lvis':
        dataset_train = args.lvis_dataset(  args.lvis_ann_path, 
                                            args.lvis_image_folder_path, 
                                            args.lvis_dino_feats_folder_path, 
                                            aug=args.aug, 
                                            strong_aug=args.strong_aug,
                                            extra_shots=args.extra_shots,
                                            support_norm=args.support_norm,
                                            train=args.training,
                                            return_mask=args.use_mask_head,
                                            mixed_support_selection = args.use_mixed_support_selection,
                                            support_feat_dim=args.support_feat_dim,
                                            val_cls_ids=args.val_cls_ids,
                                            region_detect=args.region_detect,
                                            use_text = args.use_text,
                                            )
    elif args.dataset_name == 'obj365':
        dataset_train = args.obj365_dataset(args.obj365_ann_path, 
                                            args.obj365_image_folder_path, 
                                            args.obj365_dino_feats_folder_path, 
                                            aug=args.aug, 
                                            strong_aug=args.strong_aug,
                                            extra_shots=args.extra_shots,
                                            support_norm=args.support_norm,
                                            train=args.training,
                                            return_mask=args.use_mask_head,
                                            mixed_support_selection = args.use_mixed_support_selection,
                                            support_feat_dim=args.support_feat_dim,
                                            val_cls_ids=args.val_cls_ids,
                                            region_detect=args.region_detect,
                                            use_text = args.use_text,
                                            )
    elif ',' not in args.dataset_name:
        raise NotImplementedError
    else:
        list_of_dataset_names = args.dataset_name.split(',')
        for index,name in enumerate(list_of_dataset_names):
            if index == 0:
                dataset_train=getattr(args,name+'_dataset')(   getattr(args,name+'_ann_path'), 
                                                                getattr(args,name+'_image_folder_path'), 
                                                                getattr(args,name+'_dino_feats_folder_path'), 
                                                                aug=args.aug, 
                                                                strong_aug=args.strong_aug,
                                                                extra_shots=args.extra_shots,
                                                                support_norm=args.support_norm,
                                                                train=args.training,
                                                                return_mask=args.use_mask_head,
                                                                mixed_support_selection = args.use_mixed_support_selection,
                                                                support_feat_dim=args.support_feat_dim,
                                                                val_cls_ids=args.val_cls_ids,
                                                                use_text = args.use_text,
                                                                )
            else:
                dataset_train+=getattr(args,name+'_dataset')(  getattr(args,name+'_ann_path'), 
                                                                getattr(args,name+'_image_folder_path'), 
                                                                getattr(args,name+'_dino_feats_folder_path'), 
                                                                aug=args.aug, 
                                                                strong_aug=args.strong_aug,
                                                                extra_shots=args.extra_shots,
                                                                support_norm=args.support_norm,
                                                                train=args.training,
                                                                return_mask=args.use_mask_head,
                                                                mixed_support_selection = args.use_mixed_support_selection,
                                                                support_feat_dim=args.support_feat_dim,
                                                                val_cls_ids=args.val_cls_ids,
                                                                use_text = args.use_text,
                                                                )
    

    if args.local_rank>0:
        data_loader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                collate_fn=collate_fn,#NOTE avg for text, removed for image
                num_workers=args.num_workers,
                drop_last=True,
            )
        data_loader_train = dist.warp_loader(data_loader_train,shuffle=True)
        
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)    
        data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=collate_fn,#NOTE  avg for text, removed for image
                num_workers=args.num_workers,
            )
    
    if args.resume is not None:
        model_without_ddp = load_weight(args,model_without_ddp)
    print("Start Training")
    start_time = time.time()
    AP = 0
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, 
                        data_loader_train, 
                        optimizer, 
                        args.device, 
                        epoch, 
                        weight_dict=args.weight_dict,
                        print_freq=10,
                        max_norm=args.max_norm
                        )
        if dist.is_main_process():
            AP = get_eval_res(model,shot,
                        args.test_save_path,
                        args.test_ann_path,
                        args.test_image_folder_path,
                        args.test_batch_size,
                        data_loader_test,
                        args.device,
                        epoch=epoch+1)
            if AP:
                print("cur AP30:", AP)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, f"EPOCH:{epoch}_"+args.model_name+f'-AP30:{AP}'+'.pth')
            if args.local_rank>0:
                dist.save_on_master(model_without_ddp.state_dict(), checkpoint_path)
            else:
                torch.save(model_without_ddp.state_dict(), checkpoint_path)
    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
              
if __name__ == "__main__":
    train()

    




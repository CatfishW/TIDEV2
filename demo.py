from model.tide import TIDE
from model.encoder import HybridEncoder,Tidev1Encoder
from model.decoder import TideTransformer,Tidev1Decoder
from model.segmentation import DETRsegm
from backbone.resnet import ResNet
from dataset.postprocessor import process
from config import args
import os
import torch
from misc.utils import load_weight,support_set_preprocess

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
                num_feature_levels=args.num_feature_level
                ).to(args.device)
if args.use_mask_head:
    model = DETRsegm(model, freeze_tide=(args.frozen_weights is not None)).to(args.device)
model.eval()
model = load_weight(args,model)

if __name__ == "__main__":
    classes = ["background","1","2","3","4","5","6"]#,"3","4","5","6"]#,"7","8","9"]#,'3','4','5','6']
    text = None#['fridge','chair','potted plant','window']
    #text = ['background',"black bag","yellow bag"]
    #classes = ["background","1","2"]
    #text = ['background',"person",'racket']
    #classes = ['background',"4","9"]
    #classes = ['background','doll']
    #temp = os.listdir(args.dino_feats_folder_path)
    #temp.sort(key=lambda x:int(x))
    #classes = ["background"]+temp[:]
    support_set_path = os.path.join("dataset", "support_set")#
    #support_set_path = os.path.join("dataset_project", "screw_support")#
    #support_set_path = os.path.join("dataset_project", "medical_support")
    #support_set_path = os.path.join("dataset","MaskedImages")
    query_img_path = '../street.jpg'
    #query_img_path = os.path.join("dataset", "query_set","5.jpg")
    #query_img_path = os.path.join("dataset_project", "screw_query","1745015439181746178.jpg") #  #
    #query_img_path = os.path.join("dataset_project", "medical_query","1745762354575970306.jpg")

    support_set = support_set_preprocess(classes,support_set_path)
    process(
            model=model,
            backbone_support=args.support_backbone.to(args.device),
            query_img_path=query_img_path,
            support_set=support_set,
            resize_shape=(640,640),
            POS_THRES=0.0,
            iou=0.3,
            classes=classes,
            use_fixed=False,
            debug=False,
            masked_feats=False,
            use_square_resize=True,
            square_resize_value=(0,0,0),
            text = text
            )
            

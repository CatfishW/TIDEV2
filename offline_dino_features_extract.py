from dataset import generate_support_avg_features, generate_support_features, generate_support_features_distributed
from config import args
if __name__ =="__main__":
    print("starting generating dino features...")
    generate_support_avg_features.gen_feat_by_cate(args.dino_ann_path,
                                                args.dino_feats_folder_path,
                                                args.dino_image_folder_path,
                                                args.support_backbone.to(device=args.device),
                                                poly=False,
                                                samples_per_category=30,
                                                text_mode=False,
                                                clip_preprocessor=args.clip_preprocessor,
                                                )
    # generate_dino_features.gen_feat_by_cate(args.dino_ann_path,
    #                                         args.dino_feats_folder_path,
    #                                         args.dino_image_folder_path,
    #                                         args.dino.to(device=args.device),
    #                                         poly=True,
    #                                         reshape=False,
    #                                         masked=False,
    #                                         )

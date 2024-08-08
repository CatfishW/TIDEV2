run
# python -m torch.distributed.launch --nproc_per_node=4 main.py

find number files in each folder
# find . -maxdepth 1 -type d | while read dir; do count=$(find "$dir" -type f | wc -l); echo "$dir : $count"; done    

kill all python process
# ps -ef | grep .python | awk '{print $2}'|xargs kill -9

fiftyone convert \
    --input-dir /root/fiftyone/open-images-v7/train \
    --input-type fiftyone.types.OpenImagesV7Dataset \
    --input-kwargs max_samples=100 shuffle=True \
    --output-dir /root/fiftyone/coco \
    --output-type fiftyone.types.COCODetectionDataset \
import json
p = './gt_instances.json'#'./pascal_val2012.json'
with open (p,'r') as f:
    data = json.load(f)
import pdb;pdb.set_trace()
print()
#VOC2012
# {'supercategory': 'none', 'id': 1, 'name': 'aeroplane'}
# {'supercategory': 'none', 'id': 2, 'name': 'bicycle'}
# {'supercategory': 'none', 'id': 3, 'name': 'bird'}
# {'supercategory': 'none', 'id': 4, 'name': 'boat'}
# {'supercategory': 'none', 'id': 5, 'name': 'bottle'}
# {'supercategory': 'none', 'id': 6, 'name': 'bus'}
# {'supercategory': 'none', 'id': 7, 'name': 'car'}
# {'supercategory': 'none', 'id': 8, 'name': 'cat'}
# {'supercategory': 'none', 'id': 9, 'name': 'chair'}
# {'supercategory': 'none', 'id': 10, 'name': 'cow'}
# {'supercategory': 'none', 'id': 11, 'name': 'diningtable'}
# {'supercategory': 'none', 'id': 12, 'name': 'dog'}
# {'supercategory': 'none', 'id': 13, 'name': 'horse'}
# {'supercategory': 'none', 'id': 14, 'name': 'motorbike'}
# {'supercategory': 'none', 'id': 15, 'name': 'person'}
# {'supercategory': 'none', 'id': 16, 'name': 'pottedplant'}
# {'supercategory': 'none', 'id': 17, 'name': 'sheep'}
# {'supercategory': 'none', 'id': 18, 'name': 'sofa'}
# {'supercategory': 'none', 'id': 19, 'name': 'train'}
# {'supercategory': 'none', 'id': 20, 'name': 'tvmonitor'}
#COCO
# {'supercategory': 'person', 'id': 1, 'name': 'person'}
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
# {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
# {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
# {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
# {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}
# {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}
# {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}
# {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
# {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}
# {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}
# {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}
# {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}
# {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}
# {'supercategory': 'animal', 'id': 16, 'name': 'bird'}
# {'supercategory': 'animal', 'id': 17, 'name': 'cat'}
# {'supercategory': 'animal', 'id': 18, 'name': 'dog'}
# {'supercategory': 'animal', 'id': 19, 'name': 'horse'}
# {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}
# {'supercategory': 'animal', 'id': 21, 'name': 'cow'}
# {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}
# {'supercategory': 'animal', 'id': 23, 'name': 'bear'}
# {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}
# {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}
# {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}
# {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}
# {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}
# {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}
# {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}
# {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}
# {'supercategory': 'sports', 'id': 35, 'name': 'skis'}
# {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}
# {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}
# {'supercategory': 'sports', 'id': 38, 'name': 'kite'}
# {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}
# {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}
# {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}
# {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}
# {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}
# {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}
# {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}
# {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}
# {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}
# {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}
# {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}
# {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}
# {'supercategory': 'food', 'id': 52, 'name': 'banana'}
# {'supercategory': 'food', 'id': 53, 'name': 'apple'}
# {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}
# {'supercategory': 'food', 'id': 55, 'name': 'orange'}
# {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}
# {'supercategory': 'food', 'id': 57, 'name': 'carrot'}
# {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}
# {'supercategory': 'food', 'id': 59, 'name': 'pizza'}
# {'supercategory': 'food', 'id': 60, 'name': 'donut'}
# {'supercategory': 'food', 'id': 61, 'name': 'cake'}
# {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}
# {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}
# {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}
# {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}
# {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}
# {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}
# {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}
# {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}
# {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}
# {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}
# {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}
# {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}
# {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}
# {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}
# {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}
# {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}
# {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}
# {'supercategory': 'indoor', 'id': 84, 'name': 'book'}
# {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}
# {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}
# {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}
# {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
# {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}
# {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}
vocid_2_cocoid = {
    1: 15,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72
}
["1", "2", "3", "4", "5", "6", "7", "9", "16", "17", "18", "19", "20", "21", "44", "62", "63", "64", "67", "72"]



import os.path as osp
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import tqdm
from typing import *

from utils import *

def label_for_id(id):
    gt_lis=get_gt(id)
    seg_lis=get_seg(id)
    seg_box_lis=[seg2box(seg) for seg in seg_lis]
    seg_label_lis=[]
    for gt in gt_lis:
        info={}
        # 找一个对应的seg
        gt_box=gt["bbox"]
        gt_seg_iou=0
        seg_for_gt=None
        for seg,seg_box in zip(seg_lis,seg_box_lis):
            iou=calculate_iou(gt_box,seg_box)
            if gt_seg_iou<iou:
                gt_seg_iou=iou
                seg_for_gt=seg
        info["category"]=gt["category"]
        info["bbox"]=gt_box
        info["segmentation"]=seg_for_gt
        seg_label_lis.append(info)
    return seg_label_lis


for num in range(1,11):
    ann_filename=path_utils.ann_filename=f"anns_{num}.json"
    path_utils.init()
    test_path()
    id_lis=[]
    with open(osp.join(ann_root,ann_filename),"r") as fp:
        anns=json.load(fp)
        for i in anns["images"]:
            id_lis.append(i["file_name"][:-4])

    ann_lis=[]
    for id in tqdm.tqdm(id_lis):
        seg_label_lis=label_for_id(id)
        ann_lis.append({"file_name":id+".png","segmentation":seg_label_lis})

    ann_path="ann_1221"
    with open(osp.join(ann_path,ann_filename),"w") as fp:
        json.dump(ann_lis,fp)

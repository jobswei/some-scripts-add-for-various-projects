
import os.path as osp
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import *

data_root="/home/wzy/ultralytics/data/SAMRS/images"
gt_root="/home/wzy/ultralytics/data/SAMRS/ins"
mask_root="/home/wzy/ultralytics/data/SAMRS/gray"
ann_root="/home/wzy/ultralytics/ann_1218"
ann_filename="anns_1.json"

def init():
    global ann_file
    with open(osp.join(ann_root,ann_filename),"r") as fp:
        ann_file=json.load(fp)

init()

def test_path():
    print(data_root)
    print(gt_root)
    print(mask_root)
    print(ann_root)
    print(ann_filename)
    
def data_path(id):
    return osp.join(data_root, id+".png")
def mask_path(id):
    return osp.join(mask_root, id+".png")
def gt_path(id):
    return osp.join(gt_root, id+".pkl")

def get_seg(id):
    img_lis=ann_file["images"]
    for img in img_lis:
        if img["file_name"]==id+".png":
            index=img_lis.index(img)
            break
    ann=ann_file["annotations"][index]
    assert img_lis[ann["image_id"]]["file_name"]==id+".png"
    return ann["segmentation"]
def get_gt(id):
    with open(gt_path(id),"rb") as fp:
        gt=pickle.load(fp)
    new_gt=[]
    for i in range(len(gt)):
        new_gt.append({"category":gt[i]["category"],"bbox":gt[i]["bbox"].tolist()})
    return new_gt

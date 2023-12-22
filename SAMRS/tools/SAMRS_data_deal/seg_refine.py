import os.path as osp
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import *
import tqdm
from utils import *


def inbox(point:tuple, bbox:list):
    x, y = tuple(point)
    x1, y1, x2, y2 = tuple(bbox)
    return x1 <= x <= x2 and y1 <= y <= y2

ann_path="/home/wzy/ultralytics/ann_1221"
out_path="/home/wzy/ultralytics/ann_1222"

for num in range(2,11):
    ann_file=f"anns_{num}.json"

    with open(osp.join(ann_path,ann_file),"r") as fp:
        anns=json.load(fp)


    for ann in tqdm.tqdm(anns):
        segmentations=ann["segmentation"]
        for segment in segmentations:
            bbox=segment["bbox"]
            seg=segment["segmentation"]
            if seg==None:
                segment["segmentation"]=box2ploy(bbox)
                continue
            new_seg=[]
            for i in range(0,len(seg),2):
                x=seg[i]
                y=seg[i+1]
                if inbox((x,y),bbox):
                    new_seg.append([x,y])
                else:
                    near_point=list(point_refine((x,y),bbox))
                    if not near_point in new_seg:
                        new_seg.append(near_point)
            new_seg=np.array(new_seg).flatten().tolist()
            if len(new_seg)<4:
                new_seg=box2ploy(bbox)
            segment["segmentation"]=new_seg

    with open(osp.join(out_path,ann_file),"w") as fp:
        json.dump(anns,fp)

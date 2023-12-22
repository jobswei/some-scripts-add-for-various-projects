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
from shapely.geometry import Polygon


def inbox(point:tuple, bbox:list):
    x, y = tuple(point)
    x1, y1, x2, y2 = tuple(bbox)
    return x1 <= x <= x2 and y1 <= y <= y2

ann_path="/home/wzy/ultralytics/ann_1222_refine"
out_path="/home/wzy/ultralytics/ann_1222_simplify"

for num in range(2,11):
    ann_file=f"anns_{num}.json"

    with open(osp.join(ann_path,ann_file),"r") as fp:
        anns=json.load(fp)


    for ann in tqdm.tqdm(anns):
        segmentations=ann["segmentation"]
        for segment in segmentations:
            bbox=segment["bbox"]
            seg=segment["segmentation"]

            seg=unsqueeze_seg(seg)
            seg=[tuple(i) for i in seg]
            polygon = Polygon(seg)
            
            # 简化多边形
            simplified_polygon = polygon.simplify(1, preserve_topology=True)
            
            seg=list(simplified_polygon.exterior.coords)
            seg=[list(i) for i in seg]
            seg=squeeze_seg(seg)

            segment["segmentation"]=seg

    with open(osp.join(out_path,ann_file),"w") as fp:
        json.dump(anns,fp)

import os.path as osp
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import *
from .vis_utils import *
import numpy as np
def _align_bbox(bbox):
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    # box1和box2的格式：(x_min, y_min, x_max, y_max)
    box1,box2=tuple(map(_align_bbox, [box1,box2]))
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    
    # 计算交集的坐标
    x_intersection_tl = max(x1_tl, x2_tl)
    y_intersection_tl = max(y1_tl, y2_tl)
    x_intersection_br = min(x1_br, x2_br)
    y_intersection_br = min(y1_br, y2_br)
    
    # 计算交集的宽度和高度
    intersection_width = max(0, x_intersection_br - x_intersection_tl)
    intersection_height = max(0, y_intersection_br - y_intersection_tl)
    
    # 计算交集面积
    intersection_area = intersection_width * intersection_height
    
    # 计算并集面积
    box1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    box2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def seg2box(seg):
    if type(seg[0])==float:
        seg=align_seg(seg)
    box=[seg[0][0],seg[0][1],seg[0][0],seg[0][1]]
    for points in seg:
        x1=min(points[0::2])
        y1=min(points[1::2])
        x2=max(points[0::2])
        y2=max(points[1::2])
        box[0]=min(x1,box[0])
        box[1]=min(y1,box[1])
        box[2]=max(x2,box[2])
        box[3]=max(y2,box[3])
    return box

def box2ploy(bbox):
    ploy=[(bbox[0],bbox[1]),(bbox[2],bbox[1]),(bbox[2],bbox[3]),(bbox[0],bbox[3])]
    return ploy

def point_refine(point,bbox):
    x, y = tuple(point)
    x1, y1, x2, y2 = tuple(bbox)
    res_x= x if x1 <= x <= x2 else (x1 if np.abs(x-x1)<np.abs(x-x2) else x2)
    res_y= y if y1 <= y <= y2 else (y1 if np.abs(y-y1)<np.abs(y-y2) else y2)
    return (res_x,res_y)

def inbox(point:tuple, bbox:list):
    x, y = tuple(point)
    x1, y1, x2, y2 = tuple(bbox)
    return x1 <= x <= x2 and y1 <= y <= y2
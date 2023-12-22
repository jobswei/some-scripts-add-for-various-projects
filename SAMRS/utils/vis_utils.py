import os.path as osp
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import *
from .path_utils import *
import numpy as np

def show_img(img):
    if type(img)==str:
        img=cv2.imread(data_path(id))
    plt.figure()
    plt.imshow(img)
    plt.show()
def align_seg(segmentation_coords):
    xy=[]
    for i in range(0,len(segmentation_coords),2):
        xy.append([segmentation_coords[i],segmentation_coords[i+1]])
    segmentation_coords = xy
    return segmentation_coords
def visualize_segmentation(image, segmentation_coords):
    if type(segmentation_coords[0])==float or len(segmentation_coords[0])==2:
        segmentation_coords=[segmentation_coords]

    if type(image)==str:
        image = plt.imread(image)

    # 创建画布
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # 绘制分割区域
    for seg in segmentation_coords:
        if type(seg[0])==float:
            seg=align_seg(seg)
        poly = patches.Polygon(seg, edgecolor='r', facecolor='none')
        plt.gca().add_patch(poly)

    # 显示图像和分割区域
    # plt.axis('off')
    plt.show()

def visualize_bbox(image,bbox):
    if type(bbox[0])!=list:
        bbox=[bbox]
    if type(image)==str:
        image = plt.imread(image)
    
    for box in bbox:
        x1,y1,x2,y2=tuple(map(int,box))
        image=cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 显示图像
    # cv2.imshow('image', image)
    show_img(image)

def unsqueeze_seg(seg):
    seg=np.array(seg).reshape(-1,2).tolist()
    return seg

def squeeze_seg(seg):
    seg=np.array(seg).flatten().tolist()
    return seg
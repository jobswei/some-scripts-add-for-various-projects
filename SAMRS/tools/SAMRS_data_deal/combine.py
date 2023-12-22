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

ann_path="/home/wzy/ultralytics/ann_1222_simplify"
out_path="/home/wzy/ultralytics/ann_1222_simplify"
all=[]

for num in tqdm.tqdm(range(1,11)):
    ann_file=f"anns_{num}.json"

    with open(osp.join(ann_path,ann_file),"r") as fp:
        anns=json.load(fp)


    all.extend(anns)


all.sort(key=lambda x:x["file_name"])

with open(osp.join(out_path,"annotations.json"),"w") as fp:
    json.dump(all,fp)
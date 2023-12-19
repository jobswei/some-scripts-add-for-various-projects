import time as time
import json
import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from PIL import Image

img=Image.open('data/coco8-seg/images/train/000000000009.jpg')
lis="0.782016 0.986521 0.937078 0.874167 0.957297 0.782021 0.950562 0.739333 0.825844 0.561792 0.714609 0.420229 0.657297 0.391021 0.608422 0.4 0.0303438 0.750562 0.0016875 0.811229 0.003375 0.889896 0.0320156 0.986521".split(" ")
lis=[float(lis[i])*img.width if i//2==0 else float(lis[i])*img.height for i in range(len(lis))]
anns=[{"segmentation":[lis]}]
fig, ax = plt.subplots(figsize=(img.width/100,img.height/100))
polygons = []
color = []
for ann in anns:
    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    if 'segmentation' in ann:
        if type(ann['segmentation']) == list:
            # polygon
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                # coco图片的坐标原点位于左上，plt的坐标原点位于左下。
                # 故需要对图片的Y坐标进行翻转，即图片高度减原Y值得到翻转后的Y值。
                poly[:, 1] = 334 - poly[:, 1]
                polygons.append(Polygon(poly))
                color.append(c)

colors = 100*np.random.rand(len(polygons))
p = PatchCollection(polygons, alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
fig.colorbar(p, ax=ax)
print(img)
plt.savefig('1.png')
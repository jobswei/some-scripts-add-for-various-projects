import os
import os.path as osp
import json
import tqdm

def align(img_information):
    img_information["file_name"]=img_information["filename"]
    del img_information["filename"]
    return img_information
data_root="./data/demo_video_fps2/hico2"

def change(mode):
    # 去掉没有hoi的数据
    with open(osp.join(data_root,f"annotations/{mode}_hico.json")) as fp:
        origin_lis=json.load(fp)
    print(len(origin_lis))
    lis=[]
    for info in tqdm.tqdm(origin_lis):
        if len(info["hoi_annotation"])==0:
            continue
        lis.append(info)
    print(len(lis))
    with open(osp.join(data_root,f"annotations/{mode}_hico.json"),"w") as fp:
        json.dump(lis,fp)

def change2(mode):
    # 解决标反的数据
    with open(osp.join(data_root,f"annotations/{mode}_hico.json")) as fp:
        origin_lis=json.load(fp)
    count_reverse=0
    count_error=0
    lis=[]
    for info in tqdm.tqdm(origin_lis):
        hoi_lis=info["hoi_annotation"]
        bbox_lis=info["annotations"]
        ok=True
        for hoi in hoi_lis:
            if bbox_lis[hoi["subject_id"]]["category_id"]!=1 and bbox_lis[hoi["object_id"]]["category_id"]==1:
                print(info)
                hoi["object_id"],hoi["subject_id"]=hoi["subject_id"],hoi["object_id"]
                
                count_reverse+=1
            elif bbox_lis[hoi["subject_id"]]["category_id"]==1 and bbox_lis[hoi["object_id"]]["category_id"]!=1:
                continue
            else:
                print(info["file_name"])
                count_error+=1
                ok=False
        if ok:
            lis.append(info)
    print(f"reverse data: {count_reverse}")
    print(f"error data: {count_error}")
    with open(osp.join(data_root,f"annotations/{mode}_hico.json"),"w") as fp:
        json.dump(lis,fp)
print("train")
change("trainval")
change2("trainval")
print("test")
change("test")
change2("test")

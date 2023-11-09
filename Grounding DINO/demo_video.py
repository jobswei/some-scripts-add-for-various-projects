import sys
sys.path.append("/home/wzy/GroundingDINO/")
from groundingdino.util.inference import load_model, load_image, predict, annotate,load_frame
import cv2
import os
import os.path as osp
import shutil
from tqdm import tqdm 

def del_dir(dir):
    if osp.exists(dir):
        shutil.rmtree(dir)

def demo_video2(source, output_file,TEXT_PROMPT = "person . handbag ."):
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    # TEXT_PROMPT = "person . handbag ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    #########################
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))  # 设置帧率为 30
    frame_count = 0
    frame_sampling_rate =1
    #########################

    bar=tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_sampling_rate == 0:

            image_source, image = load_frame(frame)
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            # cv2.imwrite(osp.join(osp.dirname(output_file),"sdfsdfsdfsf.jpg"), annotated_frame)
            # frame_r=cv2.imread(osp.join(osp.dirname(output_file),"sdfsdfsdfsf.jpg"))
            out.write(annotated_frame)

        frame_count += 1
        bar.update(1)
        if frame_count==100:
            break
    bar.close()
    out.release()
    cap.release()
    # os.remove(osp.join(osp.dirname(output_file),"sdfsdfsdfsf.jpg"))
source="/home/wzy/GroundingDINO/picta/videos/objects_20231020_165953.mp4"
out=f"/home/wzy/GroundingDINO/picta/predict/{osp.basename(source)}"
demo_video2(source,out)
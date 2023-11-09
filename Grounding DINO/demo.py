import sys
sys.path.append("/home/wzy/GroundingDINO/")
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import os.path as osp
import shutil
from tqdm import tqdm 

def _predict(model,infile,outfile,TEXT_PROMPT = "person . handbag ."):
    IMAGE_PATH = infile
    # TEXT_PROMPT = "person . handbag ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(outfile, annotated_frame)

def get_frame(video_path,output_dir,num=-1):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    frame_sampling_rate =1

    print("Begin to get frame!")
    bar=tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if num==-1 else num)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 保存帧为图像文件
        if frame_count % frame_sampling_rate == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
        bar.update(1)
        if num>0 and frame_count>=num:
            break
    bar.close()
    cap.release()
def get_video(frame_dir,output_video):

    # 获取帧的文件列表
    frame_files = [os.path.join(frame_dir, filename) for filename in os.listdir(frame_dir)]
    frame_files.sort()  # 确保帧按顺序排列

    # 获取第一帧的大小，以便创建视频
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # 设置视频编解码器和输出视频对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
    out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 设置帧率为 30

    # 逐帧写入视频
    for frame_file in tqdm(frame_files):
        frame = cv2.imread(frame_file)
        out.write(frame)

    # 关闭输出视频
    out.release()
def del_dir(dir):
    if osp.exists(dir):
        shutil.rmtree(dir)
def demo_video(source, output_file):
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

    frame_dir=osp.join(osp.dirname(source),"frame")
    output_frame_dir=osp.join(osp.dirname(output_file),"frame_out")
    del_dir(frame_dir)
    del_dir(output_frame_dir)
    os.mkdir(frame_dir)
    os.mkdir(output_frame_dir)


    get_frame(source,frame_dir,num=100)
    lis=os.listdir(frame_dir)
    for img in tqdm(lis):
        _predict(model,osp.join(frame_dir,img),osp.join(output_frame_dir,img))
    get_video(output_frame_dir,output_file)

    del_dir(frame_dir)
    del_dir(output_frame_dir)

source="/home/wzy/GroundingDINO/picta/videos/objects_20231020_165953.mp4"
out=f"/home/wzy/GroundingDINO/picta/predict/{osp.basename(source)}"
demo_video(source,out)
from ultralytics import YOLO

# Load a model
model = YOLO('checkpoints/yolov8s-seg.pt')  # load an official model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg',save=True)  # predict on an image
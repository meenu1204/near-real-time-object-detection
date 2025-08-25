# File: train_yolo.py

# import necessary libraries
from src.utils.config import BASE_DIR
from ultralytics import YOLO

def train_yolo_model():
    """
    Train YOLOv8n on the dataset mentioned in data.yaml
    :return: 
    """

    # yolov8n.pt: Pretrained weights on COCO
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data=str(BASE_DIR / "data.yaml"),
        epochs=10,
        imgsz=320,
        batch=4,
        workers=5,
        name="yolov8n",
        pretrained=True,
        patience=5,
        cache=False
    )

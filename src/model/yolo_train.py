# File: train_yolo.py
# Description: Trains a YOLOv8 object detector on dataset defined in data.yaml

# import necessary libraries
#from src.utils.config import BASE_DIR
from ultralytics import YOLO

def train_yolo_model():
    """
    Train the model
    """

    # yolov8n.pt: Load COCO-pretrained YOLOv8n weights
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=30,
        imgsz=320,
        batch=2,
        workers=2,
        name="yolov8n",
        pretrained=True,
        patience=10,
        cache=False
    )

if __name__ == "__main__":
    train_yolo_model()
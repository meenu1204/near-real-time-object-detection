# File: main.py

import os
from src.utils.config import (
    RAW_TRAIN_IMAGE_DIR,
    RAW_TRAIN_LABEL_DIR,
    RAW_VAL_IMAGE_DIR,
    RAW_VAL_LABEL_DIR,
    YOLO_TRAIN_IMG_DIR,
    YOLO_TRAIN_LABEL_DIR,
    YOLO_VAL_IMG_DIR,
    YOLO_VAL_LABEL_DIR
)
from src.data.data_yolo_preparation import convert_kitti_to_yolo
from src.model.yolo_train import train_yolo_model

if __name__ == "__main__":

    print("Starting YOLO training...")
    train_yolo_model()
    print("Training complete...")
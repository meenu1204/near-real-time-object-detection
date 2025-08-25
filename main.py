# File: main.py

import os
from src.data.data_yolo_preparation import convert_kitti_to_yolo
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

if __name__ == "__main__":
    os.makedirs(YOLO_TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(YOLO_TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(YOLO_VAL_IMG_DIR, exist_ok=True)
    os.makedirs(YOLO_VAL_LABEL_DIR, exist_ok=True)

    print("Loading YOLO dataset...")

    print("Converting training images to YOLO format...")
    convert_kitti_to_yolo(RAW_TRAIN_IMAGE_DIR, RAW_TRAIN_LABEL_DIR, YOLO_TRAIN_IMG_DIR, YOLO_TRAIN_LABEL_DIR )

    print("Converting validation images to YOLO format...")
    convert_kitti_to_yolo(RAW_VAL_IMAGE_DIR, RAW_VAL_LABEL_DIR, YOLO_VAL_IMG_DIR, YOLO_VAL_LABEL_DIR)

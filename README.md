# Near Real-time Object Detection

## Description
A real-time computer vision pipeline that converts KITTI dataset labels to YOLO format, trains a YOLOv8 model for pedestrian, car and cyclist detection, and streams test images via Apache Kafka to simulate live camera feeds.

## Features
- Dataset handling:
  - Uses KITTI raw images and labels
  - Splits raw data into train/val (80:20)
  - Convert KITTI labels to YOLO format (normalised 0-1) and organises files into YOLO folder structure

- Model training:
  - Trains YOLO8n with configurable hyperparameters
  - Evaluate performance on validation set
  - Tracks metrics
  - Automatically saves best.pt (YOLO model with weights from the best validation performance)

- Inference(FastAPI)
  - REST API endpoint (/predict) for object detection
  - Accepts images and return predictions with class and confidence.

## Project Setup

1. Clone the repository

```bash
git clone https://github.com/meenu1204/near-real-time-object-detection
cd near-real-time-object-detection
```

2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Dataset preparation
  - The raw KITTI dataset already has training and test folders.
  - Raw train dataset was split into train/val (80:20)
  ```bash
  python -m scripts.raw_train_val
  ```
  - Converted KITTI labels in raw tran/val dataset to YOLO format
  ```bash
  python main.py
  ```
  - Overall dataset structure:

  ```
  dataset/
  │
  ├── raw/                         # Official KITTI data
  │   ├── train/                   # training split (from KITTI "training/")
  │   │   ├── image_2/             # raw images (.png)
  │   │   └── label_2/             # KITTI labels (.txt)
  │   │
  │   ├── val/                     # validation split (20% from training)
  │   │   ├── image_2/             # raw images (.png)
  │   │   └── label_2/             # KITTI labels (.txt)
  │   │
  │   └── test/                    # official KITTI test (no labels)
  │       └── image_2/
  │
  ├── yolo/                        # YOLO-ready format (converted labels)
  │   ├── train/
  │   │   ├── images/              # training images (copied from raw/train/image_2)
  │   │   └── labels/              # YOLO labels (converted from raw/train/label_2)
  │   │
  │   └── val/
  │       ├── images/              # validation images (copied from raw/val/image_2)
  │       └── labels/              # YOLO labels (converted from raw/val/label_2)

  ```


6. cgvh
7. bjj
```bash
```

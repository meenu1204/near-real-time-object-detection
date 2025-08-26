# Near Real-time Object Detection

## Description


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

     
  - 


6. cgvh
7. bjj
```bash
```

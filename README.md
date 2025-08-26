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

- Real-time Kafka Streaming
  - Kafka Producer streams KITTI test images (as if from a live camera)
  - Kafka Consumer reads stream and runs YOLO inference
  - Live visualization (with counts of Cars, Pedestrians, and Cyclists)

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
6. Training YOLOv8 on the KITTI dataset (Since the focus was on building an end-to-end pipeline, we did not focus on improving the confidence score of model)
  - yolo_train.py

  ```bash
  from ultralytics import YOLO
  
  model = YOLO("yolov8n.pt")
  model.train(
      data=str(BASE_DIR / "data.yaml"),
      epochs=20,
      imgsz=320,
      batch=4,
      workers=5,
      name="yolov8n",
      pretrained=True,
      patience=5,
      cache=False
)
```
  - data.yaml
  ```
  train: dataset/yolo/train/images
  val: dataset/yolo/val/images
  nc: 3
  names: ["Car", "Pedestrian", "Cyclist"]

  ```
- Results of training are saved under runs/detect/yolov8n/. The best weights can be seen below:
```bash
runs/detect/yolov8n/weights/best.pt
```
7. Model serving:
- Wrap best.pt inside a FastAPI service (src/api/app.py)
- Provides a REST API (/predict) where clients can upload a test image and get predictions
- Run locally with uvicorn
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
- Test the API using a Curl command in your Terminal
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test.png"
```
8. Real-time Kaka Streaming
- Start Kafka and create topic
```bash
brew services start kafka
kafka-topics --create --topic kitti-stream --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```
- RUN producer (Sending KITTI test images)
```bash
python src/streaming/kafka_producer.py
```
- Run Kafka Consumer
```bash
python src/streaming/kafka_consumer.py
``
   


  
10. 
```bash
```

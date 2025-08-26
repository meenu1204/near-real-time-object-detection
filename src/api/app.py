# File: app.py

# Import necessary libraries
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from src.utils.config import YOLO_BEST_MODEL

# Load trained model
model = YOLO(YOLO_BEST_MODEL)
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run Inference
    results = model.predict(img)

    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            predictions.append({
            "class": model.names[cls_id],
            "confidence": round(conf,3)
            })

    return {"predictions": predictions}







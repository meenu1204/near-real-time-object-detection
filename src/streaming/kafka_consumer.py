# File: kafka_consumer.py
# Description: Receive KITTI test images  from Kafka and run YOLO inference

# necessary libraries
import cv2
import numpy as np
from confluent_kafka import Consumer
from ultralytics import YOLO
from src.utils.config import YOLO_BEST_MODEL, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, YOLO_BEST_MODEL, KAFKA_TOPIC


# Load trained model
model = YOLO(YOLO_BEST_MODEL)

# Kafka consumer
consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
    "group.id":"'kitti_group",
    "auto.offset.reset": "earliest"
})

consumer.subscribe(["kitti-stream"])

try:
    while True:
        msg = consumer.poll(3.0)
        if msg is None:
            continue

        if msg.error():
            print("Consumer error: {}".format(msg.error()))
            continue

        file_name = msg.key().decode()
        frame = cv2.imdecode(np.frombuffer(msg.value(), np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            print("Warning: Failed to decode frame")
            continue

        # Run inference
        results = model(frame)

        car_count, ped_count, cyc_count = 0, 0, 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label == "Car":
                    car_count += 1
                    color = (0, 255, 0)
                elif label == "Pedestrian":
                    ped_count += 1
                    color = (255, 0, 0)
                elif label == "Cyclist":
                    cyc_count += 1
                    color = (0, 0, 255)
                else:
                    continue

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        summary_text = f"Cars: {car_count} | Pedestrians: {ped_count} | Cyclists: {cyc_count}"
        cv2.putText(frame, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(f"KITTI car Stream - {file_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Shutting down")
finally:
    consumer.close()
    cv2.destroyAllWindows()
# File: kafka_producer.py
# Description: Simulate real-time streaming of KITTI images (test images) using Apache Kafka for inference

# necessary libraries
import os
import cv2
import glob
import time
from confluent_kafka import Producer
from src.utils.config import TEST_IMAGE_DIR, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
topic = KAFKA_TOPIC

image_paths = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png")))

for img_path in image_paths:
    img = cv2.imread(img_path)

    if img is None:
        continue

    _, buf = cv2.imencode(".png", img)
    filename = os.path.basename(img_path).replace(".png", "")

    producer.produce(topic, key=filename.encode(), value=buf.tobytes())
    producer.poll(0)

    print(f"Produced {filename} into Kafka under the topic {topic}")
    time.sleep(1)

producer.flush()

print("Finished streaming all test images")
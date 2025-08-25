# File: config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Read .env file and load all its KEY=value pairs
load_dotenv()

# Get base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
RAW_TRAIN_IMAGE_DIR = BASE_DIR / os.getenv("RAW_TRAIN_IMAGE_DIR")
RAW_TRAIN_LABEL_DIR = BASE_DIR / os.getenv("RAW_TRAIN_LABEL_DIR")

RAW_VAL_IMAGE_DIR = BASE_DIR / os.getenv("RAW_VAL_IMAGE_DIR")
RAW_VAL_LABEL_DIR = BASE_DIR / os.getenv("RAW_VAL_LABEL_DIR")

#OUT_IMAGE_DIR = BASE_DIR / os.getenv("OUT_IMAGE_DIR")
#OUT_LABEL_DIR = BASE_DIR / os.getenv("OUT_LABEL_DIR")
#TEST_IMAGE_DIR = BASE_DIR / os.getenv("TEST_IMAGE_DIR")
#TRAINED_MODEL = BASE_DIR / os.getenv("TRAINED_MODEL")

# KITTI to YOLO class mapping
#YOLO_CLASS_MAP = {
#    "Car": 0,
#    "Pedestrian": 1,
#   "Cyclist": 2
#}

# Dataset split configuration
SPLIT_TEST_SIZE = 0.2
SEED = 42
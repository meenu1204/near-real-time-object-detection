# File: train_val_split.py
# Description: Split dataset into train and val (80:20)

# Import necessary libraries
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split

from src.utils.config import (
    RAW_TRAIN_IMAGE_DIR,
    RAW_TRAIN_LABEL_DIR,
    RAW_VAL_IMAGE_DIR,
    RAW_VAL_LABEL_DIR,
    SPLIT_TEST_SIZE,
    SEED,
)
from logger_config import setup_logger

# Setting up a module-specific logger
logger = setup_logger(__name__)

def move_file(img_path: Path):
    """
    Move a single image and its corresponding label (if it exists)
    into the validation directories
    """

    img_name = img_path.name
    lbl_name = img_name.replace(".png", ".txt")

    src_lbl = RAW_TRAIN_LABEL_DIR / lbl_name

    try:
        shutil.move(img_path, RAW_VAL_IMAGE_DIR)
        if src_lbl.exists():
            shutil.move(src_lbl, RAW_VAL_LABEL_DIR)
        else:
            logger.warning(f"{lbl_name} does not exist")
    except Exception as e:
        logger.error(f"Failed to move {img_name} :{e}")

def split_dataset_train_val():
    """
    Split KITTI raw dataset (existing train images + labels) into train and validation set (80:20).
    Move 20% of train data into validation directory
    """

    raw_img_files = sorted(RAW_TRAIN_IMAGE_DIR.glob("*.png"))

    # Split into train and val
    _, val_img_files = train_test_split(
        raw_img_files,
        test_size=SPLIT_TEST_SIZE,
        shuffle=True,
        random_state=SEED
    )

    RAW_VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # Move files concurrently
    with ThreadPoolExecutor() as executor:
        executor.map(move_file, val_img_files)

if __name__ == "__main__":
    split_dataset_train_val()

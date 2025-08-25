# File: raw_train_val.py

# Import necessary libraries
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from src.utils.config import (
    RAW_TRAIN_IMAGE_DIR,
    RAW_TRAIN_LABEL_DIR,
    RAW_VAL_IMAGE_DIR,
    RAW_VAL_LABEL_DIR,
    SPLIT_TEST_SIZE,
    SEED
)

def split_dataset_train_val():
    """
    Split KITTI raw train dataset (existing images + labels) into training and validation set.
    """

    img_files = sorted(glob.glob(os.path.join(RAW_TRAIN_IMAGE_DIR, "*.png")))

    # Split into train and val in the ratio 80:20
    _, val_img_files = train_test_split(
        img_files,
        test_size=SPLIT_TEST_SIZE,
        shuffle=True,
        random_state=SEED
    )

    os.makedirs(RAW_VAL_IMAGE_DIR, exist_ok=True)
    os.makedirs(RAW_VAL_LABEL_DIR, exist_ok=True)

    # Move images and labels to validation folder
    for val_img in val_img_files:
        img_name = os.path.basename(val_img)
        lbl_name = img_name.replace(".png", ".txt")

        src_lbl = os.path.join(RAW_TRAIN_LABEL_DIR, lbl_name)
        dst_img = os.path.join(RAW_VAL_IMAGE_DIR, img_name)
        dst_lbl = os.path.join(RAW_VAL_LABEL_DIR, lbl_name)

        shutil.move(val_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print("Warning: Label not found for {dst_img}")

if __name__ == "__main__":
    split_dataset_train_val()

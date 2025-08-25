# File: convert_kitti_to_yolo.py

import os
import glob
import cv2
from src.utils.config import YOLO_CLASS_MAP

def convert_kitti_to_yolo(raw_img_dir, raw_label_dir, yolo_img_dir, yolo_label_dir):
    """
    Converts the KITTI labels to YOLO format
        - Loop through all label files (and get corresponding images) in KITTI format
        - Load images to get height and width (for normalisation purposes)
        - Read each object line in KITTI label file and convert them to YOLO format (values between 0 and 1)
        - Write image and label to YOLO format for training and val dataset
    Args:
         raw_img_dir (str): path to raw images
         raw_label_dir (str): path to raw labels
         yolo_img_dir (str): path to yolo images
         yolo_label_dir (str): path to yolo converted labels
    """

    label_file = sorted(glob.glob(os.path.join(raw_label_dir, "*.txt")))

    # Loop through all label files in KITTI dataset
    for label_file in sorted(glob.glob(os.path.join(raw_label_dir, "*.txt"))):
        try:
            img_name = os.path.basename(label_file).replace(".txt", ".png")
            img_path = os.path.join(raw_img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"Warning: {img_path} does not exist")
                continue

            # Load image for normalisation
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: {img} does not exist")
                continue

            h, w, _ = img.shape

            yolo_labels  = []
            # Read each object line in KITTI label file
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    obj_class = parts[0]

                    if obj_class not in YOLO_CLASS_MAP:
                        continue

                    xmin, ymin, xmax, ymax = map(float, parts[4:8])

                    # Check if bounding box is valid
                    if xmin < 0 or xmax > w or ymin < 0 or ymax > h or xmax <= xmin or ymax <= ymin:
                        print(f"Warning: Invalid bounding box in {label_file}")
                        continue

                    # Convert to YOLO normalised format
                    x_center = (xmin + xmax) / 2.0 / w
                    y_center = (ymin + ymax) / 2.0 / h
                    width = (xmax - xmin) / w
                    height = (ymax - ymin) / h

                    cls_id = YOLO_CLASS_MAP[obj_class]
                    yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if not yolo_labels:
                print(f"Warning: {label_file} has no valid objects")
                continue

            # Save label and image to YOLO folder
            yolo_label_file = os.path.join(yolo_label_dir, os.path.basename(label_file))
            with open(yolo_label_file, "w") as f:
                f.write("\n".join(yolo_labels))

            yolo_img_file = os.path.join(yolo_img_dir, img_name)
            if not os.path.exists(yolo_img_file):
                cv2.imwrite(yolo_img_file, img)

            print(yolo_img_file, yolo_label_file)

        except Exception as e:
            print(f"Error processing {label_file}: {e}")

    print("Converted all KITTI labels to YOLO format")





# File: convert_kitti_to_yolo.py

import os
import glob
import cv2
from src.utils.config import RAW_IMAGE_DIR, RAW_LABEL_DIR, OUT_IMAGE_DIR, OUT_LABEL_DIR, YOLO_CLASS_MAP

def convert_to_yolo(img_dir, label_dir, out_img_dir, out_label_dir):
    """
    Converts the KITTI labels to YOLO format and copy corresponding images to YOLO dataset

    Args:
    img_dir: path to KITTI images directory
    label_dir: path to KITTI labels directory
    out_img_dir: path to output directory for YOLO converted images
    out_label_dir: path to output directory for YOLO converted labels

    Returns:

    """

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for label_file in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        img_name = os.path.basename(label_file).replace(".txt", ".png")
        img_path = os.path.join(img_dir, img_name)

        print(img_path)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        yolo_labels  = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                obj_class = parts[0]

                if obj_class not in YOLO_CLASS_MAP:
                    continue

                xmin, ymin, xmax, ymax = map(float, parts[4:8])

                # Convert to YOLO format
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                cls_id = YOLO_CLASS_MAP[obj_class]
                yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        out_label_file = os.path.join(out_label_dir, os.path.basename(label_file))
        with open(out_label_file, "w") as f:
            f.write("\n".join(yolo_labels))

        out_img_file = os.path.join(out_img_dir, img_name)
        if not os.path.exists(out_img_file):
            cv2.imwrite(out_img_file, img)

    print("Converted KITTI labels to YOLO format")

if __name__ == "__main__":
    convert_to_yolo(RAW_IMAGE_DIR, RAW_LABEL_DIR, OUT_IMAGE_DIR,OUT_LABEL_DIR)




# File: convert_kitti_to_yolo.py
# Description: Convert dataset annotations from KITTI format to YOLO format

# Import necessary libraries
import cv2
from pathlib import Path
from src.utils.config import YOLO_CLASS_MAP
from logger_config import setup_logger

# Setting up a module-specific logger
logger = setup_logger(__name__)

def process_annotation(lbl_file: Path, img_file: Path ,w: int, h: int):
    """
    Parse a KITTI label file and return YOLO-format annotations.
    Args:
        lbl_file (Path): Path to the label file
        img_file (Path): Path to the image file
        w (int): Width of the image
        h (int): Height of the image
    """
    yolo_labels = []

    # Read each object line in KITTI label file
    with Path(lbl_file).open( "r", encoding="utf-8") as f:
        for lbl_line in f:
            lbl_parts = lbl_line.strip().split()
            if not lbl_parts or len(lbl_parts) not in [15]:
                continue

            cls_id = YOLO_CLASS_MAP.get( lbl_parts[0])
            if cls_id is None:
                continue

            try:
                xmin, ymin, xmax, ymax = map(float, lbl_parts[4:8])
            except ValueError:
                logger.debug(f"Non-numeric bbox {img_file}: {lbl_parts[4:8]}")
                continue

            # Check if bounding box is valid
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h or xmax <= xmin or ymax <= ymin:
                logger.debug(f"Invalid bounding box in {img_file}: {xmin},{ymin},{xmax},{ymax}")
                continue

            # Convert KITTI bounding box into the YOLO annotation format
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_labels

def kitti_to_yolo_bbox(img_file: Path, lbl_file: Path, yolo_img_dir: Path, yolo_label_dir: Path):
    """
        Converts each KITTI label + image into YOLO format
            - Load image to get height and width
            - Read each object line in KITTI label file and convert them to YOLO format (values between 0 and 1)
            - Write image and label to YOLO format for train and validation dataset
        Args:
             img_file (Path): path to raw image file
             lbl_file (Path): path to raw label file
             yolo_img_dir (Path): path to yolo images directory
             yolo_label_dir (Path): path to yolo converted labels directory
        """
    try:
        yolo_label_file = yolo_label_dir / lbl_file.name
        yolo_img_file = yolo_img_dir / (lbl_file.stem + ".png")

        # Skip if already converted
        if yolo_label_file.exists() and yolo_img_file.exists():
            logger.debug(f"Skipping {yolo_label_file}, YOLO output already exists")
            return False

        # Load image
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Image: {img_file} could not load")
            return False

        h, w, _ = img.shape
        yolo_labels = process_annotation(lbl_file, img_file, w, h)
        if not yolo_labels:
            logger.warning(f"{lbl_file} has no valid labels")
            return False

        # Save YOLO labels
        yolo_label_file.write_text("\n".join(yolo_labels))

        # Save image (Path to str for OpenCV)
        if not yolo_img_file.exists():
            cv2.imwrite(str(yolo_img_file), img)

        logger.info(f"Converted KITTI labels to YOLO format: {yolo_img_file}, {yolo_label_file}")
        return True

    except Exception as e:
        logger.error(f"Error processing {lbl_file}: {e}")
        return False


def convert_kitti_to_yolo(raw_img_dir: Path, raw_label_dir: Path, yolo_img_dir: Path, yolo_label_dir: Path):
    """
    Converts the KITTI labels to YOLO format
        - Loop through all label files (and get corresponding images) in KITTI format

    Args:
         raw_img_dir (Path): path to raw images
         raw_label_dir (Path): path to raw labels
         yolo_img_dir (Path): path to yolo images
         yolo_label_dir (Path): path to yolo converted labels
    Returns:
        bool: True if converted successfully or False if not
    """

    if not raw_img_dir.exists() or not raw_label_dir.exists():
        logger.warning(f"Input directories: {raw_img_dir} or {raw_label_dir} does not exist")
        return False

    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    # Loop through all label files in KITTI dataset
    for lbl_file in sorted(raw_label_dir.glob("*.txt")):
        img_file = raw_img_dir / f"{lbl_file.stem}.png"
        logger.debug(f"Converting Label: {lbl_file} with Image: {img_file}")
        _ = kitti_to_yolo_bbox(img_file, lbl_file, yolo_img_dir, yolo_label_dir)

    logger.info("Finished KITTI -> YOLO conversion")
    return True

# This script allow export datasets with a YOLO format
from src.process.utils import split_dataset
import fiftyone as fo
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Script to export a dataset from FiftyOne")
parser.add_argument("--name", type=str, help="Name of dataset", required=True)
parser.add_argument("--export_dir", type=str, help="Directory to export dataset from FiftyOne", required=True)
args = parser.parse_args()


# Load dataset
name = args.name
dataset = fo.load_dataset(name)

# # Export dir from args
export_dir = args.export_dir

# Export dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    split="train"
)

split_dataset(source_dir=export_dir,
              img_train_dir=export_dir+"/images/train",
              img_val_dir=export_dir+"/images/val",
              img_test_dir=export_dir+"/images/test",
              ann_train_dir=export_dir+"/labels/train",
              ann_val_dir=export_dir+"/labels/val",
              ann_test_dir=export_dir+"/labels/test")
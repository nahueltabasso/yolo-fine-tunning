# This script allow to upload a dataset to FiftyOne

import fiftyone as fo
from dotenv import load_dotenv
import os 
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Script to load a dataset to FiftyOne")
parser.add_argument("--name", type=str, help="Name of dataset", required=True)
args = parser.parse_args()

# Load environment variables
load_dotenv("../../.env")

name = args.name
ds_dir = os.path.join(os.getenv('DS_BASE_DIR_PATH'), name)
data_path = os.path.join(ds_dir, "images/train")
labels_path = os.path.join(ds_dir, "labels/train")

print("Dataset Name:", name)
print("Dataset Directory:", ds_dir)
print("Data Path:", data_path)
print("Labels Path:", labels_path)

# Define classes labels
classes = ["credit_card"] 

# Detele dataset if exists
if fo.dataset_exists(name):
    fo.delete_dataset(name)

# # Load dataset
# dataset = fo.Dataset(name=name, persistent=True)
# dataset.add_dir(
#     dataset_dir=ds_dir,
#     dataset_type=fo.types.YOLOv5Dataset,
#     split="train",
#     classes=classes
# )

# Load dataset
dataset = fo.Dataset(name=name, persistent=True)
# Iterate through splits
for split in ["train", "test", "val"]:
    data_path = os.path.join(ds_dir, f"images/{split}")
    labels_path = os.path.join(ds_dir, f"labels/{split}")
    
    print(f"Processing {split} split:")
    print(f"Data Path: {data_path}")
    print(f"Labels Path: {labels_path}")
    
    # Load split
    split_dataset = fo.Dataset.from_dir(
        dataset_dir=ds_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        classes=classes
    )
    
    # Merge with main dataset
    dataset.merge_samples(split_dataset)

# Verify classes after load dataset
print("Dataset Classes:", dataset.default_classes)

# Apply class assignment if necessary
if not dataset.default_classes:
    dataset.default_classes = classes
    for sample in dataset:
        detections = sample.ground_truth.detections
        for detection in detections:
            if detection.label == "1":
                detection.label = "credit_card"
        sample.save()

# Add metadata
dataset.compute_metadata(overwrite=True)

# Save
dataset.save()

# Verify classes again
print("Saved Dataset Classes:", dataset.default_classes)


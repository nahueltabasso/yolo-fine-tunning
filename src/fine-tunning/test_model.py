from ultralytics import YOLO
# from ultralytics.utils.metrics import ConfusionMatrix
# from ultralytics.utils.ops import xywh2xyxy
from dotenv import load_dotenv
# import torch
import numpy as np
import argparse
import os 

# Load environment variables
load_dotenv("../../.env")

# Arguments
parser = argparse.ArgumentParser(description="Script to load a dataset to FiftyOne")
parser.add_argument("--model", type=str, help="Name of Model", required=True)
parser.add_argument("--data", type=str, help="Directory to yaml", required=True)
args = parser.parse_args()

model_path = args.model
data = args.data
if model_path is None or data is None:
    model_path = os.getenv('MODEL_WEIGHT_PATH')
    data = "path_to_your_dataset.yaml"

# Load Fine-Tunning YOLO model
model = YOLO(model_path)

# Make a test
results = model.val(data=data, conf=0.25, iou=0.65)

# Extract metrics
mAP50 = results.box.map50
mAP50_95 = results.box.map

print(f"mAP@0.5: {mAP50:.4f}")
print(f"mAP@0.5:0.95: {mAP50_95:.4f}")

# Calculate precision and recall for each class
precision = results.box.p 
recall = results.box.r     

# Show precision and recall to each class
for i, (p, r) in enumerate(zip(precision, recall)):
    print(f"Class {i}: Precisión = {p:.4f}, Recall = {r:.4f}")

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1-score average: {np.nanmean(f1_score):.4f}")

# Show Confusion Matrix
if hasattr(results, 'confusion_matrix'):
    print("Confusion Matrix:")
    conf_matrix = results.confusion_matrix.matrix
    print(conf_matrix)
else:
    print("Error!")

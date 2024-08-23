from ultralytics import YOLO
from dotenv import load_dotenv
import argparse
import os
import torchvision as tv
import torch

# Load environment variables
load_dotenv("../../.env")

# Arguments
parser = argparse.ArgumentParser(description="Script to do an inference over an image")
parser.add_argument("--model", type=str, help="Name of Model", required=False)
parser.add_argument("--image", type=str, help="Image to process", required=False)
args = parser.parse_args()

model_path = args.model
img_path = args.image

print(f"BASE - {os.getenv('BASE_TEST_IMAGE_DIR')}")

if model_path is None:
    model_path = os.getenv('MODEL_WEIGHT_PATH')    
if img_path is None:
    img_path = os.path.join(os.getenv('BASE_TEST_IMAGE_DIR'), "75.jpg")

# Load a pre-trained YOLOv10n model
model = YOLO(model_path)

# Perform object detection on an image
results = model(img_path)
# Apply NMS to delete a bad detections
boxes_xyxy = results[0].boxes.xyxy.cpu()
clss = results[0].boxes.cls.cpu()
filtered_id_boxes = tv.ops.nms(boxes_xyxy, clss, iou_threshold=0.5)
boxes_xyxy = boxes_xyxy[filtered_id_boxes]
clss = clss[filtered_id_boxes]

print(f"BOXES - {boxes_xyxy}")
print(f"CLSS - {clss}")

# Display the results
results[0].show()

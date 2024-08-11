from ultralytics import YOLO
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Script to load a dataset to FiftyOne")
parser.add_argument("--model", type=str, help="Name of Model", required=True)
parser.add_argument("--data", type=str, help="Directory to yaml", required=True)
args = parser.parse_args()

print(f"Model -> {args.model}")
print(f"Dataset -> {args.data}")

# Load YOLOv10n model from scratch
model_name = args.model
model = YOLO(model_name)

# Train the model
data = args.data
model.train(data=data, epochs=100, imgsz=640)

# Ejecutar script
# python src/fine-tunning/train.py --model /Users/nahueltabasso/Documents/Python/yolo_fine_tunning/models/yolov10n.pt --data /Users/nahueltabasso/Documents/Python/yolo_fine_tunning/data/fo-dataset-240728/dataset.yaml


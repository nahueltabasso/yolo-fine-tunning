from ultralytics import YOLO
from dotenv import load_dotenv
import argparse
import os
import cv2

# Load environment variables
load_dotenv("../../.env")

# Arguments
parser = argparse.ArgumentParser(description="Script to do an inference over a video")
parser.add_argument("--model", type=str, help="Path to Model weights", required=False)
parser.add_argument("--video", type=str, help="Path to video file", required=False)
args = parser.parse_args()

model_path = args.model
video_path = args.video

if model_path is None:
    model_path = os.getenv('MODEL_WEIGHT_PATH')    
if video_path is None:
    video_path = os.path.join(os.getenv('BASE_TEST_VIDEO_DIR'), "IMG_2678.mov")

# Load the YOLOv8 model
model = YOLO(model_path)

# Open the video file
video = cv2.VideoCapture(video_path)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()

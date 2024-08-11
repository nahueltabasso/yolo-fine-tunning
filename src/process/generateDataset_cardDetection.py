# This script has the functionality to generate datasets 
# with the format requested by YOLO for later use 

# %%
# Import libraries
from groundingdino.util.inference import Model
from torchvision.ops import box_convert
from dotenv import load_dotenv
import cv2
import os
import torch
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np

# Load environment variables
load_dotenv("../../.env")

# %%
def resize_image(image, target_size=(640, 640)):
    # Get the original image dimensions
    h, w = image.shape[:2]
    
    # Calculate the scaling factor
    scale = min(target_size[0] / w, target_size[1] / h)
    
    # Calculate new dimensions
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate offsets for centering
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def build_dataset(model, images: list, count: int, img_output: str, ann_output: str, show: bool=False):
    """This method generate a dataset labelling by GroundingDINO with YOLO format

    Args:
        model (_type_): Instance of GroundingDINO model
        images (list): List of images to process
        count (int): Count of images and annotated files
        img_output (str): Output dir to images
        ann_output (str): Output dir to annotated files
        show (bool, optional): _description_. Defaults to False.
    """
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    TEXT_PROMPT = ["credit card"]

    for i in images:
        img = cv2.imread(filename=i)
        if img.shape != (640, 640, 3):
            img = resize_image(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if show:
            plt.imshow(img_rgb)
            plt.title("Image before labelling by GroundingDINO")
            plt.show()

        detections = model.predict_with_classes(image=img,
                                                classes=TEXT_PROMPT,
                                                box_threshold=BOX_THRESHOLD,
                                                text_threshold=TEXT_THRESHOLD)

        box_annotator = sv.BoundingBoxAnnotator()
        annotated_image = box_annotator.annotate(scene=img.copy(),
                                            detections=detections)

        if show:
            plt.imshow(annotated_image)
            plt.title("Image after inference")
            plt.show()
            
        # Convert bounding boxes to cx cy w h format
        cxcywh = box_convert(boxes=torch.Tensor(detections.xyxy),
                            in_fmt="xyxy",
                            out_fmt="cxcywh")
        
        height, width, _ = img.shape
        cxcywh_rel = cxcywh.clone()
        cxcywh_rel[:, 0] /= width
        cxcywh_rel[:, 1] /= height
        cxcywh_rel[:, 2] /= width
        cxcywh_rel[:, 3] /= height 
        
        output_path = os.path.join(img_output, str(count)+".jpg")
        cv2.imwrite(output_path, img)
   
        ann_output_path = os.path.join(ann_output, str(count)+".txt")
        with open(ann_output_path, "w") as file:
            for coords in cxcywh_rel:
                row = f"1 {coords[0]} {coords[1]} {coords[2]} {coords[3]}\n"
                file.write(row)
        
        count += 1

# %%
if __name__ == '__main__':
    # Config paths
    MODEL_CONFIG_PATH = "./GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "../../weigths/groundingdino_swint_ogc.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(model_config_path=MODEL_CONFIG_PATH,
                  model_checkpoint_path=WEIGHTS_PATH,
                  device=device)

    # Base dirs to work
    BASE_IMAGE_DIR = os.getenv('BASE_INPUT_DIR_IMAGES')
    BASE_OUTPUT_DIR_IMAGES = os.getenv('BASE_OUTPUT_DIR_IMAGES')
    BASE_OUTPUT_DIR_ANN = os.getenv('BASE_OUTPUT_DIR_ANN')

    IMAGES = os.listdir(BASE_IMAGE_DIR)

    IMAGES = [os.path.join(BASE_IMAGE_DIR, i) for i in IMAGES if i != '.DS_Store']
    IMAGE_COUNT = 1
    build_dataset(model=model,
                  images=IMAGES,
                  count=IMAGE_COUNT,
                  img_output=BASE_OUTPUT_DIR_IMAGES,
                  ann_output=BASE_OUTPUT_DIR_ANN,
                  show=True)


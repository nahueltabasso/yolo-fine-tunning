# %%
from ultralytics import YOLO
from dotenv import load_dotenv
from src.process.utils import crop_image
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv

load_dotenv("../../.env")

def extract_payment_network_by_image(model:any, 
                                     input_dir:str, 
                                     output_dir: str, 
                                     show: bool):
    desired_size = (128, 128)
    PAYMENT_NETWORK_ID = 3.0
    
    cards_path = os.listdir(input_dir)
    cards_path = [os.path.join(input_dir, c) for c in cards_path]
    
    # cards_path = cards_path[:2]
    for card in cards_path:
        print(f"Card - {card}")
        # Detect the cards elements with YOLO
        results = model(card)
        xyxy = results[0].boxes.xyxy.cpu()
        cls_ids = results[0].boxes.cls.cpu()
        confs = results[0].boxes.conf.cpu()
        
        # Apply NMS
        filtered_index = tv.ops.nms(xyxy, confs, iou_threshold=0.5)
        xyxy = xyxy[filtered_index].tolist()
        cls_ids = cls_ids[filtered_index].tolist()
        confs = confs[filtered_index]        
        # Seek the index of payment network id
        # payment_network_index = cls_ids.index(PAYMENT_NETWORK_ID)
        payment_network_indexes = np.where(np.isclose(cls_ids, PAYMENT_NETWORK_ID))[0]
        if len(payment_network_indexes) > 0:
            payment_network_index = payment_network_indexes[0]
            xyxy = xyxy[payment_network_index]
            img = cv2.imread(filename=card)
            img = crop_image(img=img, coords=xyxy)
            resized_img = cv2.resize(img, desired_size, interpolation=cv2.INTER_AREA)

            filename = card.split("/")[-1]
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)
            

# %%
# Process
INPUT_DIR = "/opt/project/data/cards"
OUTPUT_DIR = "/opt/project/data/payment-networks"
model = YOLO(os.getenv('YOLO_ELEMENTS_DETECTION_PATH'))

extract_payment_network_by_image(model=model,
                                 input_dir=INPUT_DIR,
                                 output_dir=OUTPUT_DIR,
                                 show=False)


# %% 

# Redimensionar data extra para american y cabal
OUTPUT_DIR = "/opt/project/data/payment-networks/Extra/extra-cabal"
images = os.listdir(OUTPUT_DIR)
images = [os.path.join(OUTPUT_DIR, i) for i in images]
desired_size = (128, 128)

cont = 1049
for i in images:
    # print(i)
    # print("\n\n")
    img = cv2.imread(i)
    # img = cv2.resize(img, desired_size, interpolation=cv2.INTER_AREA)

    print(f"{i} - {img.shape}")
    # cv2.imwrite(os.path.join(OUTPUT_DIR, str(cont)+".jpg"), img)
    # cont += 1
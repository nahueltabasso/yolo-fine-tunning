# To this task, We will use a public dataset so with this script we are going to 
# join this public dataset together with the images collected by me, for that 
# we are going to move the images and annotation files of the public dataset 
# together with the images and files collected by me to a new directory formatted as YOLO

# %%
# Import libraries
from dotenv import load_dotenv
from generateDataset_cardDetection import resize_image
import os
import shutil as sh
import cv2

# Load environment variables
load_dotenv("../../.env")

# Filename count
global index
index = 0

# %%
def get_ann_filename(filename):
    filename = filename.split("/")[-1]
    
    if filename.endswith(".jpeg"):
        return filename[:len(filename)-5]
    else:
        return filename[:len(filename)-4]

def move_files(input_dir: str, output_dir: str, split:str):
    global index
    print(f"INDEX - {index}")
    print(f"Input Directory - {input_dir}/images")
    img_files = os.listdir(input_dir+"/images")
    print(f"Total files in input directory - {len(img_files)}")
    
    for img_file in img_files:
        # Copy image to new folder
        img_input_path = os.path.join(input_dir+"/images", img_file)
        img_output_path = os.path.join(output_dir+"/images", split, str(index)+".jpg")

        img = cv2.imread(img_input_path)
        print(f"Shape - {img.shape}")
        if img.shape != (640, 640, 3):
            print("Entra")
            img = resize_image(image=img, target_size=(640, 640))
            cv2.imwrite(img_output_path, img)
        else:
            sh.copy2(img_input_path, img_output_path)

        # Verified if exist an ann.txt for each image
        ann_txt_name = get_ann_filename(img_file) + ".txt"
        ann_input_path = os.path.join(input_dir+"/labels", ann_txt_name)
        ann_output_path = os.path.join(output_dir+"/labels", split, str(index)+".txt")
        
        if os.path.exists(ann_input_path):
            sh.copy2(ann_input_path, ann_output_path)
        else:
            with open(ann_output_path, 'w') as ann_file:
                pass  # Empty file
        
        print(f"Img source: {img_input_path}")
        print(f"Ann source: {ann_input_path}")
        print(f"Img new dir: {img_output_path}")
        print(f"Ann new dir: {ann_output_path}\n")
        
        index += 1
    print(f"INDEX DESPUES DEL FOR {index}")

                
# %%
# Base paths
input_dir = os.getenv("BASE_INPUT_DS") + "/train"
output_dir = os.getenv("BASE_OUTPUT_DS")

# Move train files
move_files(input_dir, output_dir, "train")
# Move validation files
input_dir = os.getenv("BASE_INPUT_DS") + "/valid"
move_files(input_dir, output_dir, "val")
# Move test files
input_dir = os.getenv("BASE_INPUT_DS") + "/test"
move_files(input_dir, output_dir, "test")




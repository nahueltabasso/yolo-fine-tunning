import os
import shutil
import random
import numpy as np 

def move_files(source_dir, file_list, img_destination, ann_destination):
    """
    Moves image and annotation files from the source directory to the specified destination directories.
    
    Parameters:
    - source_dir (str): The base directory where the images and annotations are originally located.
    - file_list (list of str): List of filenames to be moved, without directory information.
    - img_destination (str): Directory where the image files should be moved.
    - ann_destination (str): Directory where the annotation files should be moved.
    
    This function constructs the full paths for the image and annotation files based on the provided
    source directory and file list. It then moves each image file to the 'img_destination' and each
    corresponding annotation file to the 'ann_destination'. The function assumes that image files are 
    in the 'source_dir/images/train' directory and annotation files are in the 'source_dir/labels/train' directory.
    
    If an annotation file does not exist for a given image, only the image file is moved.
    
    Print statements are included to show the source and destination paths of each moved file.
    
    Parameters:
    - source_dir (str): The directory containing the 'images/train' and 'labels/train' subdirectories.
    - file_list (list): A list of image filenames to be moved.
    - img_destination (str): The directory to move image files to.
    - ann_destination (str): The directory to move annotation files to.
    
    Returns:
    - None
    """
    for f in file_list:
        image_source = os.path.join(source_dir+"/images/train", f)
        label_source = os.path.join(source_dir+"/labels/train", f.rsplit('.', 1)[0] + '.txt')
        print(f"IMG_SOURCE {image_source}")
        print(f"ANN_SOURCE {label_source}")
        print(f"IMG_DEST {os.path.join(img_destination, f)}")
        print(f"ANN_DEST {os.path.join(ann_destination, f.rsplit('.', 1)[0] + '.txt')}")   
        print("\n")
        shutil.move(image_source, os.path.join(img_destination, f))
        if os.path.exists(label_source):
            shutil.move(label_source, os.path.join(ann_destination, f.rsplit('.', 1)[0] + '.txt'))


def split_dataset(source_dir: str, 
                  img_train_dir: str, 
                  img_val_dir: str, 
                  img_test_dir: str,
                  ann_train_dir: str,
                  ann_val_dir: str,
                  ann_test_dir: str,
                  train_ratio=0.8, 
                  val_ratio=0.1, 
                  test_ratio=0.1):
    """
    Splits a dataset into training, validation, and test sets and moves the respective files into the specified directories.

    Parameters:
    - source_dir (str): The base directory where the images and annotations are originally located.
    - img_train_dir (str): Directory where the training image files should be moved.
    - img_val_dir (str): Directory where the validation image files should be moved.
    - img_test_dir (str): Directory where the test image files should be moved.
    - ann_train_dir (str): Directory where the training annotation files should be moved.
    - ann_val_dir (str): Directory where the validation annotation files should be moved.
    - ann_test_dir (str): Directory where the test annotation files should be moved.
    - train_ratio (float, optional): Proportion of the dataset to be used for training. Default is 0.8.
    - val_ratio (float, optional): Proportion of the dataset to be used for validation. Default is 0.1.
    - test_ratio (float, optional): Proportion of the dataset to be used for testing. Default is 0.1.

    This function first ensures that the given train, validation, and test ratios sum up to 1. It then creates the necessary directories 
    if they do not already exist. After collecting and shuffling the list of image files, it splits them into training, validation, 
    and test sets based on the provided ratios. Finally, it moves the files into their corresponding directories using the `move_files` function.

    NOTE: This function assumes that the corresponding annotation files are in the 'source_dir/labels/train' directory and have the same base name 
    as the image files with a '.txt' extension. Images are assumed to be in 'source_dir/images/train'.

    Returns:
    - None
    """
    
    # Create directories if not exist
    for d in [img_train_dir, img_val_dir, img_test_dir, ann_train_dir, ann_val_dir, ann_test_dir]:
        os.makedirs(d, exist_ok=True)
        
    images_files = [i for i in os.listdir(img_train_dir) if i.endswith(('.jpg', '.jpeg', '.png'))]
    # Shuffle files randomly
    random.shuffle(images_files)
    
    # Calculate the number of images to each set
    num_images = len(images_files)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    
    # Split the images list
    train_files = images_files[:num_train]
    val_files = images_files[num_train:num_train + num_val]
    test_files = images_files[num_train + num_val:]
    
    # Move files to their new directories
    move_files(source_dir=source_dir,
               file_list=val_files, 
               img_destination=img_val_dir,
               ann_destination=ann_val_dir)
    move_files(source_dir=source_dir, 
               file_list=test_files, 
               img_destination=img_test_dir,
               ann_destination=ann_test_dir)
    
def crop_image(img: np.ndarray, coords: list):
    # Cast coords to int 
    x1, y1, x2, y2 = map(int, coords)
    # Make sure the coordinates are 
    # within the image boundaries.
    height, width = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    crop_image = img[y1:y2, x1:x2]
    return crop_image
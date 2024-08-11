import os
import shutil
import random

def move_files(source_dir, file_list, img_destination, ann_destination):
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
    
    # Verified that the proportions be 1
    assert train_ratio + val_ratio + test_ratio == 1.0
    
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
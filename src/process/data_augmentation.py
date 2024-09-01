# %%
from dotenv import load_dotenv
import numpy as np
import os
import cv2
import albumentations as A

load_dotenv("../../.env")

def get_output_path(path, index):
    """
    Generate a full file path for saving an image with a numeric filename.
    
    Args:
        path (str): The directory path where the image will be saved.
        index (int): The numeric index to be used as the image's filename.
    
    Returns:
        str: The complete file path with the image filename in the format "{index}.jpg".
    """
    return os.path.join(path, f"{index}.jpg")

def rotate_image(image, limit=(10, 90)):
    """
    Rotate the given image by a random angle within the specified limit.
    
    Args:
        image (numpy.ndarray): The input image to be rotated.
        limit (tuple): A tuple specifying the range of angles (in degrees) from 
                       which a random angle will be chosen for the rotation.
                       
    Returns:
        numpy.ndarray: The rotated image.
    """
    transform = A.Rotate(limit=limit, p=1.0)
    return transform(image=image)['image']

def to_grayscale(image):
    """
    Convert the given image to grayscale.
    
    Args:
        image (numpy.ndarray): The input color image to be converted to grayscale.
    
    Returns:
        numpy.ndarray: The grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def horizontal_flip(image):
    """
    Flip the given image across the horizontal axis.

    Args:
        image (numpy.ndarray): The input image to be flipped.
    
    Returns:
        numpy.ndarray: The horizontally flipped image.
    """
    return np.flip(image, axis=1)

def change_contrast(image, alpha=3.0, beta=0):
    """
    Change the contrast of the given image by adjusting its intensity levels.

    Args:
        image (numpy.ndarray): The input image whose contrast is to be changed.
        alpha (float): Contrast control. Must be a positive value. A value greater
                       than 1 will increase the contrast, while a value between 0 and 1 
                       will decrease the contrast. Default value is 3.0.
        beta (int): Brightness control. This value will be added to the pixel values 
                    after adjusting the contrast. It can be positive or negative. Default
                    value is 0.

    Returns:
        numpy.ndarray: The image with adjusted contrast.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, var=10):
    """
    Add Gaussian noise to the given image.

    Args:
        image (numpy.ndarray): The input image to which Gaussian noise will be added.
        mean (float): The mean of the Gaussian noise distribution. Default value is 0.
        var (int): The variance (squared standard deviation) of the Gaussian noise distribution. Default value is 10.

    Returns:
        numpy.ndarray: The image with added Gaussian noise.
    """
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    return cv2.add(image, gauss)

def apply_perspective_transform(image):
    """
    Apply a perspective transform to the given image.

    Args:
        image (numpy.ndarray): The input image to be transformed.

    Returns:
        numpy.ndarray: The image with a perspective transform applied.
    
    The function calculates a perspective transform matrix from source points, 
    which are the corners of the original image, to destination points, which 
    create a simulated 3D effect by shifting the image corners to new locations. 
    It then applies this matrix to the input image using the warpPerspective method 
    from OpenCV.

    The specific transformation applied shifts the corners in an asymmetric way 
    to create the effect of a change in perspective:
        - Top-left corner moves to 10% of the width and 20% of the height.
        - Top-right corner moves to 90% of the width and 10% of the height.
        - Bottom-left corner moves to 20% of the width and 80% of the height.
        - Bottom-right corner moves to 80% of the width and 90% of the height.
    """
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_points = np.float32([[width * 0.1, height * 0.2], 
                             [width * 0.9, height * 0.1], 
                             [width * 0.2, height * 0.8], 
                             [width * 0.8, height * 0.9]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (width, height))

def save_image(output_path, image):
    cv2.imwrite(output_path, image)

def augment_images(images, augment_fn, count, limit=None):
    """
    Apply a given augmentation function to a list of images and save the augmented images.

    Args:
        images (list): A list of file paths to the input images.
        augment_fn (function): The augmentation function to be applied to each image. 
                               This function should take a single argument, the image, 
                               and return the augmented image.
        count (int): The starting index for naming the saved augmented images.
        limit (int, optional): The maximum number of images to process. If None, all images 
                               in the list are processed. Default is None.

    Returns:
        int: The updated count after processing the images, which can be used as the starting 
             index for subsequent augmentations.
    """
    for image_path in images[:limit]:
        img = cv2.imread(image_path)
        augmented_image = augment_fn(img)
        output_path = get_output_path(WORK_PATH, count)
        save_image(output_path, augmented_image)
        count += 1
    return count

# %%
WORK_PATH = os.getenv("DATA_AUGMENTATION_WORK_DIR")
images = sorted([os.path.join(WORK_PATH, img) for img in os.listdir(WORK_PATH)], key=lambda x: int(os.path.basename(x).split('.')[0]))
COUNT = int(os.path.basename(images[-1]).split(".")[0]) + 1
print(f"Initial COUNT: {COUNT}")

# %%

# Define augmentation configurations
augmentations = [
    (rotate_image, None),
    (to_grayscale, None),
    (horizontal_flip, 25),
    (change_contrast, 10),
    (add_gaussian_noise, 25),
    (apply_perspective_transform, 25)
]

# Apply augmentations
for augment_fn, limit in augmentations:
    COUNT = augment_images(images, augment_fn, COUNT, limit)
    print(f"COUNT after {augment_fn.__name__} --- {COUNT}")


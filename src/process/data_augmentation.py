# %%
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import albumentations as A

load_dotenv("../../.env")

def get_ouput_path(path, index):
    output_path = path + "/" + str(index) + ".jpg"
    return output_path

# %%
WORK_PATH = "/opt/project/data/payment-networks/Cabal"
images = os.listdir(WORK_PATH)
images = sorted(images, key=lambda x: int(x.split('.')[0]))

# Init count
max_index = images[len(images)-1]
max_index = int(max_index.split(".")[0]) + 1
COUNT = max_index
print(COUNT)

images = [os.path.join(WORK_PATH, i) for i in images]
# %%
# First data augmentation technique: 
# rotating the images by a random number between 10 and 90
# images = images[:5]
for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)
    
    # Rotation transformed
    transform = A.Rotate(limit=(10, 90), p=1.0)

    # Apply transformation
    augmented = transform(image=img)
    augmented_image = augmented['image']

    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, augmented_image)
    COUNT += 1
    
# %% 
print(f"COUNT after first technique --- {COUNT}")

# %%
# Second data augmentation techique:
# convert to gray scale
for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, img)
    COUNT += 1

# %%
print(f"COUNT after second technique --- {COUNT}")

# %%
# Third data augmentation technique:
# horizontal flip
images = os.listdir(WORK_PATH)
images = [os.path.join(WORK_PATH, i) for i in images]
# For this case, use only the first 25 images
images = images[:25]

for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)

    # Apply horizontal flip
    img = np.flip(img, axis=1)
    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, img)
    COUNT += 1
    
# %%
print(f"COUNT after third technique --- {COUNT}")

# %%
# Four data augmentation techique:
# contrast change
images = os.listdir(WORK_PATH)
images = [os.path.join(WORK_PATH, i) for i in images]
# For this case, use only the first 25 images
images = images[:10]

for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)

    # alpha: Contrast gain factor (1.0 means no change)
    # beta: Brightness factor (0 means no change)
    alpha=3.0
    beta=0
    adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, img)
    COUNT += 1

# %%
print(f"COUNT after four technique --- {COUNT}")

# %%
# Five data augmentation technique:
# adding a noice
def add_gaussian_noise(image, mean=0, var=10):
    """
    """
    # Calculate the standard deviation from the variance
    sigma = var ** 0.5
    # Create a noise array with the same shape as the image
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    # Add noise to the original image
    noisy_image = cv2.add(image, gauss)
    return noisy_image

images = os.listdir(WORK_PATH)
images = [os.path.join(WORK_PATH, i) for i in images]
# For this case, use only the first 25 images
images = images[:25]

for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)
    img = add_gaussian_noise(img, mean=0, var=2)
    
    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, img)
    COUNT += 1


# Six data augmentation technique:
# perspective changes
def apply_perspective_transform(image):
    """
        Apply a perspective transformation to a image

        :param image: input image.
        :return: transformed image.
    """
    # Shape
    height, width = image.shape[:2]

    # Define origin points (four corners of image)
    src_points = np.float32([
        [0, 0],              # Top left corner
        [width - 1, 0],      # Top right corner
        [0, height - 1],     # Bottom left corner
        [width - 1, height - 1]  # Bottom right corner
    ])

    # Define the target points (shifted to distort perspective)
    dst_points = np.float32([
        [width * 0.1, height * 0.2],  
        [width * 0.9, height * 0.1],
        [width * 0.2, height * 0.8],
        [width * 0.8, height * 0.9]
    ])

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply perspective transformation
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    return transformed_image

images = os.listdir(WORK_PATH)
images = [os.path.join(WORK_PATH, i) for i in images]
# For this case, use only the first 40 images
images = images[:25]

# COUNT = 1214

for image in images:
    print(f"Input Image {image}")
    img = cv2.imread(filename=image)
    img = apply_perspective_transform(img)
    
    output_path = get_ouput_path(WORK_PATH, COUNT)
    print(f"Output Image {output_path}")
    cv2.imwrite(output_path, img)
    COUNT += 1



"""
This script is used to create a train-val-test split of the dataset
"""
import shutil
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from config import *
from sklearn.model_selection import train_test_split

# Define the new project directory structure and create directories if necessary
# Here, MURA_AUGMENTED_TVT is the name of the new directory that would be created
if not os.path.exists(MURA_AUGMENTED_TVT):
    # Main directory
    os.makedirs(MURA_AUGMENTED_TVT)
    # Train-Val-Test directory
    dirs = [os.path.join(MURA_AUGMENTED_TVT, "train"), os.path.join(MURA_AUGMENTED_TVT, "val"),
            os.path.join(MURA_AUGMENTED_TVT, "test")]
    for dir_name in dirs:
        os.makedirs(dir_name)
        os.makedirs(os.path.join(dir_name, "images"))
        os.makedirs(os.path.join(dir_name, "masks"))
else:
    print(f"Directory already exits at {MURA_AUGMENTED_TVT}")


# Load all images from the current directory. Update the images and masks folder paths as necessary
all_image_paths = sorted(glob(os.path.join(DATA_DIR, "**/images/*.png")))
all_mask_paths = sorted(glob(os.path.join(DATA_DIR, "**/masks/*.png")))

# isolate train-val-test image paths
train_images, val_test_images, train_masks, val_test_masks = train_test_split(all_image_paths, all_mask_paths,
                                                                              train_size=0.8, test_size=0.2,
                                                                              shuffle=True)
val_images, test_images, val_masks, test_masks = train_test_split(val_test_images, val_test_masks,
                                                                  train_size=0.5, test_size=0.5, shuffle=True)

# copying the train image-mask pairs as it is
for img_path, mask_path in tqdm(zip(train_images, train_masks), total=len(train_images)):
    shutil.copy(img_path, os.path.join(MURA_AUGMENTED_TVT, "train", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(MURA_AUGMENTED_TVT, "train", "masks", os.path.basename(mask_path)))

# copying the val image-mask pairs as it is
for img_path, mask_path in tqdm(zip(val_images, val_masks), total=len(val_images)):
    shutil.copy(img_path, os.path.join(MURA_AUGMENTED_TVT, "val", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(MURA_AUGMENTED_TVT, "val", "masks", os.path.basename(mask_path)))

# copying the test image-mask pairs as it is
for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
    shutil.copy(img_path, os.path.join(MURA_AUGMENTED_TVT, "test", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(MURA_AUGMENTED_TVT, "test", "masks", os.path.basename(mask_path)))

print("Images transferred successfully")
print("MURA Augmented Train-Val-Test (MURA_Pure_TVT) Dataset ready!")

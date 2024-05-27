import os
import torch

# Change to local paths of your local data folder
DATA_DIR = '/Users/ajay/Documents/Internships/IIT_KGP/Data/MURA_Pure_TVT'
MURA_AUGMENTED_TVT = '/Users/ajay/Documents/Internships/IIT_KGP/Data/MURA_Pure_TVT'
CHECKPOINT_DIR = '/Users/ajay/Documents/Work_in_US/Git_Projects/PDRUNet-PyTorch/models'

# Place the data having the structure as described in README.md
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Number of channels expected in the input image and output masks
IN_CHANNELS, OUT_CHANNELS = 1, 1
# Can be changed as per requirements
IMG_HEIGHT, IMG_WIDTH = 512, 512
# Dataset prepared using this split. Can be changed according to your needs
SPLIT = {"train": 80, "val": 10, "test": 10}

NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

NUM_FILTERS = 40

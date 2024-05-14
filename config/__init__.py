import os
import torch

# Change to local paths of your local data folder
DATA_DIR = 'path/to/data/dir'
CHECKPOINT_DIR = 'path/to/model/checkpoints/dir'

# Place the data having the structure as described in README.md
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of channels expected in the input image and output masks
IN_CHANNELS, OUT_CHANNELS = 1, 1
# Can be changed as per requirements
IMG_HEIGHT, IMG_WIDTH = 513, 512

NUM_EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

NUM_FILTERS = 40

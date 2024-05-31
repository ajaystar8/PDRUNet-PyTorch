"""
Trains a PyTorch semantic segmentation model using device-agnostic code.
"""

import torch.nn as nn
import torch.optim as optim

import torchvision.transforms.v2 as transforms
import wandb
from metrics import dice_coefficient

try:
    from torchmetrics.classification import BinaryPrecision, BinaryRecall
except ImportError as e:
    print("Failed to import torchmetrics. Please install it using `pip install torchmetrics`")

from config import *
from config import private_keys
from engine import train, test_model
from model_builder import PDRUNet
from data_setup import create_dataloaders
from utils.utils import get_model_summary, plot_loss_accuracy_curves

# setup wandb
wandb.login(key=private_keys.WANDB_API_Key)

# Transforms to convert the image in the format expected by the model
simple_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# Generate dataloaders
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    transform=simple_transforms,
    batch_size=BATCH_SIZE
)

# create model instance
model = PDRUNet(in_channels=IN_CHANNELS, num_filters=NUM_FILTERS, out_channels=OUT_CHANNELS).to(DEVICE)

# get_model_summary(baseline_0)

# create a loss function instance
loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer instance
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Custom created function to calculate dice score
dice_fn = dice_coefficient

# torchmetrics instances to calculate precision and recall
precision_fn = BinaryPrecision().to(DEVICE)
recall_fn = BinaryRecall().to(DEVICE)

"""
[INFO]: Change the following names for easy tracking of experiments
"""
RUN_NAME = "FDS-PDR-UNet-Metal"
MODEL_CKPT_NAME = "pdrunet.pth"

config = {
    "image_size": (IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH),
    "dataset": "MURA-Pure",
    "sample_size": len(train_dataloader) + len(val_dataloader) + len(test_dataloader),
    "train_val_test_split": "80:10:10",
    "epochs": NUM_EPOCHS,
    "batch_size": 1,
    "model": model.__class__.__name__,
    "learning_rate": 1e-5,
    "loss_fn": loss_fn.__class__.__name__,
    "optimizer": optimizer.__class__.__name__
}

# initialize a wandb run
run = wandb.init(
    project="PDR_UNet",
    name=RUN_NAME,
    config=config,
    notes="Using Metal GPUs",
    tags=["FDS", "pure", "metal"]
)

# define metrics
wandb.define_metric("train_dice", summary="max")
wandb.define_metric("val_dice", summary="max")

wandb.define_metric("train_precision", summary="max")
wandb.define_metric("val_precision", summary="max")

wandb.define_metric("train_recall", summary="max")
wandb.define_metric("val_recall", summary="max")

# copy your config
experiment_config = wandb.config

# For tracking gradients
wandb.watch(model, log="gradients", log_freq=1)

# training
wandb.alert(
    title="Training started",
    text=RUN_NAME,
    level=wandb.AlertLevel.INFO,
)

# Perform model training
baseline_0_train_results = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    dice_fn=dice_fn, precision_fn=precision_fn, recall_fn=recall_fn, model_ckpt_name=MODEL_CKPT_NAME
)

# Perform testing on the trained model
baseline_0_results = test_model(
    model_ckpt_name=MODEL_CKPT_NAME,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    dice_fn=dice_fn, precision_fn=precision_fn, recall_fn=recall_fn
)

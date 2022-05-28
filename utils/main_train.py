from tqdm import tqdm
import numpy as np
import os
import random
import glob
import re
from pathlib import Path

import cv2
import geopandas as gpd
import pandas as pd

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import model
import metrics
import config
import utis
import train_val_test

from matplotlib import pyplot as plt, cm

from torch.utils.data import DataLoader, random_split
from dataset import HistoricalImagesDataset

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb


# Dataset 
# Training loop
# classes
classes = ('no_vegetation', 'vegetation')

# Initialize model
unet = model.unet_model.to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

opt = optim.Adam(unet.parameters(), lr=config.LR)
# opt = optim.SGD(unet.parameters(), lr=config.LR, momentum=0.95, weight_decay=0.01)
scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=10, verbose=True)

# initialize a dictionary to store TRAINING history (keep track on training)
training_history = {"avg_train_loss": [], "train_accuracy": [], "IoU":[],"f1score":[], "avgDice":[]}

# # initialize a dictionary to store VALIDATION history (keep track on VALIDATION)
validation_history = {"avg_val_loss": [], "val_accuracy": [], "IoU_val":[], "f1score_val":[]}

# Using log="all" log histograms of parameter values in addition to gradients
# wandb.watch(unet, log="all")

# Autocasting 
scaler = GradScaler()

# initialize best accuracy
best_accuracy = 0.0
print(f'''Training the network for {config.NUM_EPOCHS} epochs, with a batch size of {config.BATCH_SIZE}''') # try with logger

# loop = tqdm(range(config.NUM_EPOCHS))
for e in range(config.NUM_EPOCHS):
    trained = train_val_test.train(unet, train_dataloader, opt, lossFunc, epoch=e, scaler=scaler, training_history)
    validated = train_val_test.validation(unet, val_dataloader, lossFunc, epoch=e, validation_history)
    # scheduler.step(validated['IoU_val'][-1])
    
    # Save best model
    if validated['IoU_val'][-1] > best_accuracy and e > 10: # maybe add a minimum number of epochs as conditions
        utis.save_best_model(unet, BEST_MODEL, validated, e, data_portion, rate_of_coarse_labels=n_patches)
        best_accuracy = validation_history['IoU_val'][-1]

        
        # Testing if it works 
if __name__ == '__main__':
    # Define the filters 
    filters = {'non_veg_idxs' : geo_df.query("status == 0")['index'],
           'veg_idxs' :  geo_df.query("status == 1")['index'], 
           'mixed': geo_df.query("status == 2")['index'], 
           'single_trees_idx' : geo_df.query("status == 3")['index'], 
           'hedgerows' : geo_df.query("status == 4")['index']}

    # Define transformations
    val_transform = A.Compose(
        [A.Normalize(mean=(0.512), std=(0.167)),
        ToTensorV2()])

    train_transform = A.Compose([
          A.Rotate(limit=40,p=0.9, border_mode=cv2.BORDER_CONSTANT), # p stands for the probability with which the transformations are applied
          A.HorizontalFlip(p=0.9),
          A.VerticalFlip(p=0.9), 
          A.Transpose(p=0.9),
          A.CLAHE(p=1),
          A.Normalize(mean=(0.512), std=(0.167)),
          ToTensorV2()])
    
    # Autocasting 
    scaler = GradScaler()
    
    # calling cross validation
    kfold_cross_validation(5, 
                           filters, 
                           val_transform, 
                           train_transform, 
                           save_path = 'best_model/fine_sizes', 
                           scaler=scaler)

    # one_path = 'best_model/coarse_sizes/20'
    # print(glob.glob(one_path + '/*'))
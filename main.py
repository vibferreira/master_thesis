''' Run and save the Semantic Segmetation DL models '''
''' Modify the config file for PATHS, DL model parameters etc. '''

import glob
import os

from scripts import get_patches
from scripts import get_GT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils import config
from utils import dataloader
from utils import model
from utils import metrics
from utils import config
from utils import utis
from utils import train_val_test
import wandb


##### Pre-Processing ##### 
# Decide if images need to be patched (e.g if case changes have been made to the .gpkg annotation file) 
if config.patchfying: # define in the config file if you want to patchfy the image or not
    # Get binary mask from geopackage (only necessary if the image is not in the folder already)
    gt = get_GT.Get_Ground_Truth(config.GRID_PATH, config.MASKS_GPKG_PATH, config.DEST_PATH)
    gt.get_items()

    # Get Patches from binary mask and from the image (only necessary if the image is not in the folder already)
    images = get_patches.GetPatches(config.img_paths[0], config.PATCHES_IMAGES_PATH, (256, 256))
    masks = get_patches.GetPatches(config.mask_paths[0], config.PATCHES_MASK_PATH, (256, 256))
    images.get_items()
    masks.get_items()

##### Data Split and Dataloader ##### 
# I could have create a class here for this, improve it later!!!!
train_dataloader = dataloader.train_dataloader
val_dataloader = dataloader.val_dataloader
test_dataloader = dataloader.test_dataloader

##### DL model (training and validation loop) ##### 
# classes
classes = ('no_vegetation', 'vegetation')

# Initialize model
unet = model.unet_model.to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

opt = optim.Adam(unet.parameters(), lr=config.LR)
# opt = optim.SGD(unet.parameters(), lr=config.LR, momentum=0.95, weight_decay=0.01)
# scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=10, verbose=True)
scheduler = StepLR(opt, step_size=20, gamma=0.1)

# initialize a dictionary to store TRAINING history (keep track on training)
training_history = {"avg_train_loss": [], "train_accuracy": [], "IoU":[],"f1score":[], "avgDice":[]}

# # initialize a dictionary to store VALIDATION history (keep track on VALIDATION)
validation_history = {"avg_val_loss": [], "val_accuracy": [], "IoU_val":[], "f1score_val":[]}

# Using log="all" log histograms of parameter values in addition to gradients
wandb.login()
wandb.init(project="my-awesome-project")
wandb.watch(unet, log="all")

# Autocasting 
scaler = GradScaler()

# initialize best accuracy
best_accuracy = 0.0
print(f'''Training the network for {config.NUM_EPOCHS} epochs, with a batch size of {config.BATCH_SIZE}''') # try with logger

# loop = tqdm(range(config.NUM_EPOCHS))
iter_ = 0
data_portion = 'all_coarse_labels' # ['all_coarse_labels','coarse_plus_fine_labels', 'fine_labels', 'coarse_labels']
n_patches = 283
for e in range(config.NUM_EPOCHS):
    trained = train_val_test.train(unet, train_dataloader, opt, lossFunc, epoch=e, scaler=scaler, training_history=training_history)
    validated = train_val_test.validation(unet, val_dataloader, lossFunc, epoch=e, validation_history=validation_history)
    scheduler.step()
    
    # Save best model
    if validated['IoU_val'][-1] > best_accuracy: #and e > 10: # maybe add a minimum number of epochs as conditions
        utis.save_best_model(unet, config.BEST_MODEL, validated, e, data_portion, rate_of_coarse_labels=n_patches) 
        best_accuracy = validation_history['IoU_val'][-1]







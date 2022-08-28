''' This file stores the configurations and parameter settings
Implementation based on https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/'''

# other options --> yaml, json, create a class in which the variables would be attributes

import glob
import torch

# =========================================================== #
# PATHS (IMAGES, MASKS, DESTINATION FOLDER)
# =========================================================== #
# Geting Ground truth 
MASKS_GPKG_PATH = 'data/geopackages/UPDATED_6_mask_per_year.gpkg'
GRID_PATH = 'data/geopackages/grid.gpkg'
DEST_PATH = 'data/masks'

# Getting patches
MASKS = 'data/masks'
IMAGES = 'data/images'
PATCHES_IMAGES_PATH = 'data/patches/images'
PATCHES_MASK_PATH = 'data/patches/masks'
img_paths = glob.glob(IMAGES +'/*.tif')
mask_paths = glob.glob(MASKS +'/*.tif')

# Everything else
IMAGES_PATH = 'data/patches/images/1942'
MASK_PATH = 'data/patches/masks/1942'
BEST_MODEL = 'best_model/unet'

FILTER_PATH = 'data/geopackages/patch_keys.gpkg'
TILE_PATH = 'data/geopackages/tiles.gpkg'
TEST_DATASET_PATH = '../data/test_dataset'

image_paths = glob.glob(IMAGES_PATH +'/*.tif')
mask_paths = glob.glob(MASK_PATH +'/*.tif')

# =========================================================== #
# PATCHFY - Decide if the images need to be patchyfied or not 
# =========================================================== #
patchfying = False

# =========================================================== #
# SET WHICH DEVICE TO USE
# =========================================================== #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

# =========================================================== #
# DATA SPLIT - decide if its a geographycal split (tile) or based on the quality of the dataset (filter)
# =========================================================== #
split = 'tile' # 'filter'

# =========================================================== #
# DATASET AND DATALOADER
# =========================================================== #
PATCH_SIZE = 256

# =========================================================== #
# MODEL
# =========================================================== #
BACKBONE = "mobilenet_v2"
N_CHANNELS = 1
N_CLASSES = 1

# =========================================================== #
# TRAINING AND VALIDATION LOOPS
# =========================================================== #
LR = 0.001 # learning rate
NUM_EPOCHS = 10
BATCH_SIZE = 16

# =========================================================== #
# Attention
# =========================================================== #
ATT = 'scse' #or'scse' Attention module used in decoder of the model. Available options are **None** and **scse** (https://arxiv.org/abs/1808.08127)
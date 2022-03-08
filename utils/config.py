''' This file stores the configurations and parameter settings
Implementation based on https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/'''

# other options --> yaml, json, create a class in which the variables would be attributes

import glob
import torch

# =========================================================== #
# PATHS (IMAGES, MASKS, DESTINATION FOLDER)
# =========================================================== #

IMAGES_PATH = r'data\patches\images\1942'
MASK_PATH = r'data\patches\masks\1942'

image_paths = glob.glob(IMAGES_PATH +'\*.tif')
mask_paths = glob.glob(MASK_PATH +'\*.tif')

# =========================================================== #
# SET WHICH DEVICE TO USE
# =========================================================== #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================== #
# DATASET AND DATALOADER
# =========================================================== #
PATCH_SIZE = 256
# SPLIT = 

# =========================================================== #
# MODEL
# =========================================================== #
BACKBONE = "mobilenet_v2"
N_CHANNELS = 1
N_CLASSES = 1

# =========================================================== #
# TRAINING AND VALIDATION LOOPS
# =========================================================== #
LR = 0.0001 # learning rate
NUM_EPOCHS = 5
BATCH_SIZE = 4


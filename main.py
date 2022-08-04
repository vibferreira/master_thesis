''' Run and save the Semantic Segmetation DL models '''
import glob
import os

from scripts import get_patches
from scripts import get_GT

from utils import config
from utils import split

##### Pre-Processing ##### 
# Decide if images need to be patched (e.g if case changes have been made to the .gpkg annotation file) 
if config.patchfying: # define in the config file if you want to patchfy the image or not
    # Get binary mask from geopackage (only necessary if the image is not in the folder already)
    gt = get_GT.Get_Ground_Truth(config.GRID_PATH, config.MASKS_GPKG_PATH, config.DEST_PATH)
    print(gt.get_items())

    # Get Patches from binary mask and from the image (only necessary if the image is not in the folder already)
    images = get_patches.GetPatches(config.img_paths[0], config.PATCHES_IMAGES_PATH, (256, 256))
    masks = get_patches.GetPatches(config.mask_paths[0], config.PATCHES_MASK_PATH, (256, 256))
    images.get_items()
    masks.get_items()

##### Data Split and Data Loader ##### 

# separate file 


##### DL model ##### 





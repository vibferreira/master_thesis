''' 
This script permforms 
3) Patchizing individual images and masks 
3) Save the patches (preserving the spatial information)
'''

# MAY REQUIRE RESIZING THE IMAGE TO GET EXACT NUMBER OF PATCHES --> CONFIRM IF REALLY NEEDED

from importlib.resources import path
from pathlib import Path
import rasterio as rio
import cv2
import os

import numpy as np
import glob

# All this data paths may be given in a main python file later? 

MASKS_PATH = r'data\mask'
IMAGES_PATH = r'data\clipped_images'
PATCHES_PATH =r'data\patches'

img_paths = glob.glob(IMAGES_PATH +'\*.tif')
mask_paths = glob.glob(MASKS_PATH +'\*.tif')
patches_paths = glob.glob(PATCHES_PATH)

# print(patches_paths)

class Patchying:
    hey_I_am_learning = 'I am here just by learning purposes' # class variable shared by all instances (global variable)

    def __init__(self, img_path:str, patch_path: str) -> None:
        # Path to where to store the.tif files 
        self.img_path = img_path # class variable unique to each instance (local variable)
        self.patch_path = patch_path

    def get_file_name(self):
        ''' Get the image file name'''
        return self.img_path.split('\\')[-1].split('.')[0]

    def check_if_path_exists(self):
        ''' Check wheter the specified path exist or not, if not, created one'''
        path_Exist = os.path.exists(self.img_path)
        print('Path exists!')
        return path_Exist

new_instance = Patchying(img_paths[0], PATCHES_PATH)

print(new_instance.img_path)
print(new_instance.get_file_name())


    # First think is to check if he directory exists 


# # get file name 
# file_name = get_file_name(img_paths)

# # get_file_name(img_paths)

# with rio.open(img_paths[0]) as src:
#     ras_data = src.read().astype('uint8')
#     ras_meta = src.profile

# # make any necessary changes to raster properties:
# ras_meta['year'] = file_name[0]

# print(ras_meta['year'])

# Get Patches

# Caculate number of patches and return an array to store the new origins

# Save the patches with the new origins 
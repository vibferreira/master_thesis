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
from patchify import patchify

import os
import numpy as np
import glob

# All this data paths may be given in a main python file later? 
MASKS_PATH = r'data\mask'
IMAGES_PATH = r'data\clipped_images'
PATCHES_PATH =r'data\patches'

img_paths = glob.glob(IMAGES_PATH +'\*.tif')
mask_paths = glob.glob(MASKS_PATH +'\*.tif')
patches_paths = glob.glob(PATCHES_PATH + r'\1942')

# print(os.path.exists(PATCHES_PATH + r'\1942'))
# print(patches_paths)

class GetPatches:
    current_path = Path.cwd() # class variable shared by all instances (global variable), here just for learning purposes

    def __init__(self, img_path:str, patch_path: str) -> None:
        '''Args: 
        img_path: .tif file path
        patch_path: destination folder of the patches'''
        self.img_path = img_path # class variable unique to each instance (local variable)
        self.patch_path = patch_path

    def get_file_name(self) -> str:
        ''' Get the image file name'''
        return self.img_path.split('\\')[-1].split('.')[0]

    def check_if_path_exists(self):
        ''' Check wheter the specified path exist or not, if not, created one named with the file name'''
        file_name = self.get_file_name()
        path = f'{self.patch_path}\{file_name}'
        pathExists = os.path.exists(f'{self.patch_path}\{file_name}')
        if not pathExists:
            os.makedirs(path)
            return print(f'Creating folder {file_name}')
        else:
            return print(f'Folder already exists')

    def get_patches(self, img:np.array, patch_size: tuple):
        ''' Return image patches
        Args:
        img: image in numpy array format
        patch_size: tuple representing the target patch size'''
        return patchify(img, patch_size, step=patch_size[0])

new_instance = GetPatches(img_paths[0], PATCHES_PATH)

# print(new_instance.img_path)

with rio.open(img_paths[0]) as src:
    ras_data = src.read().astype('uint8')
    ras_meta = src.profile

# make any necessary changes to raster properties, e.g.:
ras_meta['year'] = '1942'
# print(patchify(ras_data[0],(512,512)), step=512))

print(new_instance.get_patches(ras_data[0], (512,512)).shape)

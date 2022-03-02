''' 
The GetPaches class
3) Patchize INDIVIDUAL images 
3) Save the patches (preserving the spatial information)
'''
# MAY REQUIRE RESIZING THE IMAGE TO GET EXACT NUMBER OF PATCHES --> CONFIRM IF REALLY NEEDED

from importlib.resources import path
from pathlib import Path

import rasterio as rio
import cv2
from patchify import patchify
from rasterio.crs import CRS
from rasterio.transform import from_origin

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

class GetPatches:
    current_path = Path.cwd() # class variable shared by all instances (global variable), here just for learning purposes

    def __init__(self, img_path:str, patch_path: str, patch_size:tuple) -> None:
        '''Args: 
        img_path: .tif file path
        patch_path: destination folder of the patches'''
        self.img_path = img_path # class variable unique to each instance (local variable)
        self.patch_path = patch_path
        self.patch_size = patch_size

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

    def get_top_left_bounds_of_patches(self):
        file_name = self.get_file_name()

        # read the image 
        with rio.open(img_paths[0]) as src:
            ras_data = src.read().astype('uint8')
            ras_meta = src.profile # get the original image profile

            # make any necessary changes to raster properties, e.g.:
            ras_meta['year'] = file_name

        # get patches
        patches = self.get_patches(ras_data[0], self.patch_size)

        # Total number of patches
        patch_size = np.concatenate(patches).shape[0]

        # Array to store the new top left bounds
        n_patches = np.zeros((patch_size, 2))

        rows = patches.shape[0]
        cols = patches.shape[1]

        i, p, z = 0, 0, 0
        for x in range(rows):
            x += z # update X after concluding all columns for one row
            for y in range(cols):
                # get the position in spatial coords
                new_x, new_y = src.xy(x, y + i)
                # assign the spatial coords to an empty array
                n_patches[p] = new_x, new_y 
            
                i+=(self.patch_size[0]-1)  # number of pixels to add
                p+=1  # adding up to total number of patches

            z += (self.patch_size[0]-1) # to update X (go down -255 pixels after each row)
            i = 0 # set to zero again after reading each row

        return n_patches

    def save_patches_with_crs(self):

        pass

    def get_item(self):
        file_name = self.get_file_name()

        # read the image 
        with rio.open(img_paths[0]) as src:
            ras_data = src.read().astype('uint8')
            ras_meta = src.profile # get the original image profile

            # make any necessary changes to raster properties, e.g.:
            ras_meta['year'] = file_name

        # n_patches = self.get_top_left_bounds_of_patches(ras_data[0])

        # Check if path exists 
        self.check_if_path_exists()

        # save patches
        self.save_patches_with_crs()
        pass


new_instance = GetPatches(img_paths[0], PATCHES_PATH, (512, 512))

# print(new_instance.img_path)

with rio.open(img_paths[0]) as src:
    ras_data = src.read().astype('uint8')
    ras_meta = src.profile

# make any necessary changes to raster properties, e.g.:
ras_meta['year'] = '1942'
# print(patchify(ras_data[0],(512,512)), step=512))
print(ras_meta['crs'])

print(new_instance.get_top_left_bounds_of_patches())

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

class GetPatches:
    current_path = Path.cwd() # class variable shared by all instances (global variable), here just for learning purposes

    def __init__(self, 
                img_path:str, 
                patch_path: str, 
                patch_size:tuple) -> None:
        '''
        Args: 
        img_path: .tif file path
        patch_path: destination folder of the patches
        patch_size: tuple, define patch size e.g (512 X 512)
        Returns:
        None
        '''
        self.img_path = img_path # class variable unique to each instance (local variable)
        self.patch_path = patch_path
        self.patch_size = patch_size

    def get_file_name(self) -> str:
        ''' Get the image file name
        Args:
        self.img_path: image path name
        Returns:
        str: file name'''
        return self.img_path.split('\\')[-1].split('.')[0]

    def check_if_path_exists(self) -> str:
        ''' Check wheter the specified path exist or not, if not, created new path named with the file name
        Args:
        self.get_file_name(): func to get the file_name
        self.patch_path: folder where to save the patches
        
        Returns:
        path: str, path where to save the patches 
        '''
        file_name = self.get_file_name()
        path = f'{self.patch_path}\{file_name}'
        pathExists = os.path.exists(f'{self.patch_path}\{file_name}')
        if not pathExists:
            os.makedirs(path)
            print(f'Creating folder {path}')
            return path
        else:
            print(f'Folder already exists {path}')
            return path

    def get_patches(self, img:np.array) -> np.array:
        ''' Return image patches
        Args:
        img: image in numpy array format
        patch_size: tuple representing the target patch size
        Returns:
        np.array: patched image with shape [row_original_image, col_original_image, row_patch, col_patch] '''
        return patchify(img, self.patch_size, step=self.patch_size[0])

    def get_top_left_bounds_of_patches(self, patches: np.array, src:rio) -> np.array:
        ''' Get top left bounds coords for each patch and stores in a np.array
        Args:
        img: np.array, image that will be patched
        Returns:
        src: rasterio image object 
        n_patches = zero like numpy array with shape  [number_of_patches, 2] (to stores new x and y)
        '''
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

    def save_patches_with_crs(self, 
                            patches: np.array, 
                            n_patches:np.array, 
                            path:str, 
                            ras_meta:dict,
                            ras_data:np.array, 
                            number_bands:int=1) -> None:

        '''Save patches in given folder
        Args:
        patches: np.array, patches from image 
        n_patches: np.array, contains the new X and Y coords
        path: str, path where to save the images
        ras_meta: dictionary, metadata of rasterio image
        ras_data: np.array, image
        number_bands: number of bands of original image
        Returns:
        None'''

        crs = CRS.from_epsg('2154')
        patches = np.concatenate(patches)
        print('Saving Images')
        for i in range(n_patches.shape[0]):
            # transform 
            transform = from_origin(n_patches[i][0], n_patches[i][1], 0.75, 0.75) # output size
            with rio.open(f"{path}\{ras_meta['year']}_{i}.tif", 'w',
                    driver='GTiff',
                    height=self.patch_size[0],
                    width=self.patch_size[1],
                    dtype=ras_data.dtype,
                    count=number_bands, # number of bands, CAREFUL if the image has RGB
                    crs = crs, 
                    transform=transform) as dst:
                    dst.write(patches[i,np.newaxis,:,:]) # add a new axis, required by rasterio 
            

    def get_items(self):
        file_name = self.get_file_name()

        # read the image 
        with rio.open(self.img_path) as src:
            ras_data = src.read().astype('uint8')
            ras_meta = src.profile # get the original image profile

            # make any necessary changes to raster properties, e.g.:
            ras_meta['year'] = file_name

        # Check if path exists 
        path_patches = self.check_if_path_exists()

        # Get patches
        patches = self.get_patches(ras_data[0])

        # New array storinf X and Y for each patch 
        n_patches = self.get_top_left_bounds_of_patches(patches, src)

        # save patches
        self.save_patches_with_crs(patches, n_patches, path_patches, ras_meta ,ras_data, number_bands=1)


if __name__ == '__main__':
    # All this data paths may be given in a main python file later? 
    MASKS_PATH = r'data\mask'
    IMAGES_PATH = r'data\clipped_images'
    PATCHES_IMAGES_PATH = r'data\patches\images'
    PATCHES_MASK_PATH = r'data\patches\masks'

    img_paths = glob.glob(IMAGES_PATH +'\*.tif')
    mask_paths = glob.glob(MASKS_PATH +'\*.tif')
    
    # Example saving image from 1942
    images = GetPatches(img_paths[0], PATCHES_IMAGES_PATH, (512, 512))
    masks = GetPatches(mask_paths[0], PATCHES_MASK_PATH, (512, 512))

    images.get_items()
    masks.get_items()


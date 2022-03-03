''' 
This script performs 
1) Clipping masks by grid extent (based on the extent of ground truth from 2020);
2) Clipping the Historical Aerial Images into the grid extent;
4) Rasterizing ground truth into a binary mask. 
'''

import re
import glob
from os import path
import numpy as np
from pandas import DataFrame

import geopandas as gpd
import cv2
import fiona
import rioxarray as rxr
import xarray as xr
from shapely.geometry import mapping
from geocube.api.core import make_geocube

# Define the paths of the images, mask layers and grid
MASKS_PATH = 'data\mask_per_year.gpkg'
GRID_PATH = 'data\grid.gpkg'
IMAGES_PATH = 'data\images'

grid = gpd.read_file(GRID_PATH)
years = fiona.listlayers(MASKS_PATH)[1:] # list layers/years exclude last year, 2020

# read the masks 
mask_per_year = {key: gpd.read_file(MASKS_PATH, layer=key) for key in years}

# add a 1 column later binary rasterization
for value in mask_per_year.values():
    value['value'] = 1

# CRS assertion 
for year in years:
    assert mask_per_year[year].crs == grid.crs, 'CRS must match'
assert grid.is_valid.all(), 'Invalid geometry'

# clip by grid extent 
mask_per_year = {key: values.clip(grid) for key, values in mask_per_year.items()}

# Clip images by grid extent 
paths = glob.glob(IMAGES_PATH +'/*')
print(f'Number of images in the folder: {len(paths)}')

def get_xarrays():
    ''' Returns list of xarrays named by year '''
    imgs_by_year = []
    print('Getting xarrays')
    for imgs in paths:
        # get the year name from the file path 
        year = re.findall(r"[\w+']+", imgs)[-2]
        print('Done for', year)
        # read rasters as xarrays
        xr_imgs = rxr.open_rasterio(imgs).squeeze().astype(np.uint8)

        # naming the xarrays by year
        xr_imgs.name = year

        # store list of xarrays
        imgs_by_year.append(xr_imgs)

    return imgs_by_year

def clip_raster_to_mask_extent():
    imgs_by_year = get_xarrays()

    for xarray in imgs_by_year:
        # Save clipped raster
        if not path.exists(f"data/clipped_images/{xarray.name}.tif"):
            print('clipping image from', xarray.name)
            # clip by grid extent 
            xr_imgs_clipped = xarray.rio.clip(grid.geometry.apply(mapping), grid.crs)

            # save geotif
            xr_imgs_clipped.rio.to_raster(f'data/clipped_images/{xarray.name}.tif')

        else:
            print(f'{xarray.name} is already clipped!')

# Get rasterized binary mask
def rasterize_vect(df:DataFrame) -> xr:
    """ 
    Returns a binary raster mask (xarray) from a vetor layer (Geodataframe)
    """
    return make_geocube(vector_data=df,
    measurements=["value"],
    resolution=(0.75,-0.75),
    fill=0).astype(np.uint8)

def saving_binary_mask(): 
    '''Saves the binary masks into .tif files'''
    for year in years:
        if not path.exists(f'data/mask/{year}.tif'):
            print(f'Rasterizing year {year}')
            dataset = rasterize_vect(mask_per_year[year])
            out_grid_2 = dataset.to_array()
            out_grid_2.rio.to_raster(f'data/mask/{year}.tif')
        else:
            print(f'{year} is already rasterized!')

if __name__ == '__main__':
    clip_raster_to_mask_extent()
    saving_binary_mask()








 











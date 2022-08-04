''' 
This script performs 
1) Clipping the Historical Aerial Images into the grid extent (based on the extent of ground truth from 2020);
Code equivalent to clip raster by extent on QGIS, but scalable to clip several rasters;
'''
import re
import glob
from os import path
import numpy as np

import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import mapping
from geocube.api.core import make_geocube

# Define the paths of the images, mask layers and grid
MASKS_PATH = 'data\geopackages\mask_per_year.gpkg'
GRID_PATH = 'data\geopackages\grid.gpkg'
IMAGES_PATH = 'data\images'

grid = gpd.read_file(GRID_PATH)
# # print(grid)

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

# Test
if __name__ == '__main__':
    clip_raster_to_mask_extent()







 











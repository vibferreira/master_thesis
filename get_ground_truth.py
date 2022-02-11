import re
import glob
from os import path
from datetime import datetime
import numpy as np

import rasterio.features
import geopandas as gpd
import fiona
import rioxarray as rxr
import xarray as xr

from shapely.geometry import mapping

# paths 
MASKS_PATH = 'data\mask_per_year.gpkg'
IMAGES_PATH = 'data\images'

# read the masks and clip files to grid extent 
grid = gpd.read_file('data\grid.gpkg')
years = fiona.listlayers(MASKS_PATH) # list layers/years
mask_per_year = {key: gpd.read_file(MASKS_PATH, layer=key).clip(grid) for key in years} # MAY NEED TO HANDLE INVALID GEOMETRY EXCEPTION

# Clip images by grid extent 
paths = glob.glob(IMAGES_PATH +'/*')

imgs_by_year = []

for imgs in paths:
    # get the year name in the file path 
    year = re.findall(r"[\w+']+", imgs)[-2]
    xr_imgs = rxr.open_rasterio(imgs).squeeze()
    print(xr_imgs)

    # set time dim/attributes 
    xr_imgs['time'] = datetime.strptime(year, '%Y')
    xr_imgs.attrs['year'] = year

    # clip by grid extent 
    xr_imgs_clipped = xr_imgs.rio.clip(grid.geometry.apply(mapping), grid.crs)
    imgs_by_year.append(xr_imgs_clipped)

    # save the clipped image
    file_name = xr_imgs_clipped.attrs['year']
    
    if not path.exists(f"data/clipped_images/{file_name}.tif"):
        xr_imgs_clipped.rio.to_raster(f'data/clipped_images/{file_name}.tif')

# Get rasterized binary mask




 











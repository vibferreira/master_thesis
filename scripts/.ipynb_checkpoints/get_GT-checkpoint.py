''' This script performs 
1) Clipping masks by grid extent (based on the extent of ground truth from 2020);
2) Rasterizing ground truth into a binary mask. 
'''
from os import path
import numpy as np

import geopandas as gpd
import fiona
import xarray as xr
from shapely.geometry import mapping
from geocube.api.core import make_geocube

class Get_Ground_Truth:
    def __init__(self, 
                GRID_PATH:str, 
                MASKS_PATH: str, 
                DEST_PATH: str) -> None:
        
        ''' Preprocess ground truth before patchying.
        The input mask, in geopackage format, is clipped by the extent of the common grid to all images. 
        In the sequence, the geopackage is rasterize into a binary raster
        Args: 
        GRID_PATH (str): grid path
        MASKS_PATH (str): masks path
        DEST_PATH (str): destination folder of the BINARY RASTER''' 
        
        self.GRID_PATH = GRID_PATH
        self.MASKS_PATH = MASKS_PATH
        self.DEST_PATH = DEST_PATH
    
    def clip_grid_extent(self, grid:gpd.GeoDataFrame, mask_per_year:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # clip by grid extent 
        return {key: values.clip(grid) for key, values in mask_per_year.items()}
        
    # Get rasterized binary mask
    def rasterize_vect(self, df:gpd.GeoDataFrame) -> xr:
        """ Returns a binary raster mask (xarray) from a vetor layer (Geodataframe)"""
        return make_geocube(vector_data=df,
        measurements=["value"],
        resolution=(0.75, -0.75),
        fill=0).astype(np.uint8)

    def saving_binary_mask(self, years:list, mask_per_year:gpd.GeoDataFrame) -> None: 
        '''Saves the binary masks into .tif files'''
        for year in years:
            if not path.exists(f'{self.DEST_PATH}/{year}.tif'): # this should be passed to the class as well 
                print(f'Rasterizing year {year}')
                dataset = self.rasterize_vect(mask_per_year[year])
                out_grid_2 = dataset.to_array()
                out_grid_2.rio.to_raster(f'{self.DEST_PATH}/{year}.tif')
                print(f' Image is rasterized in the folder {self.DEST_PATH}')
            else:
                print(f'{year} is already rasterized!')
                
    def get_items(self) -> None:
        grid = gpd.read_file(self.GRID_PATH)
        years = fiona.listlayers(self.MASKS_PATH) # list of years 

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
        mask_per_year = self.clip_grid_extent(grid, mask_per_year)
        
        # saving 
        self.saving_binary_mask(years, mask_per_year)

# Testing if it works 
if __name__ == '__main__':
    # Define the paths of the images, mask layers and grid
    MASKS_PATH = 'data/geopackages/mask_per_year.gpkg'
    GRID_PATH = 'data/geopackages/grid.gpkg'
    DEST_PATH = 'data/masks'

    gt = Get_Ground_Truth(GRID_PATH, MASKS_PATH, DEST_PATH)
    print(gt.get_items())

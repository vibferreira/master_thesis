''' Should include all the necessary info to run the model'''

import glob
import os

from scripts import get_patches
from scripts import get_GT
from utils import config

# Get binary mask from geopackage (only necessary if the image is not in the folder already)
gt = get_GT.Get_Ground_Truth(config.GRID_PATH, config.MASKS_GPKG_PATH, config.DEST_PATH)
print(gt.get_items())

# Delete the temp files 
[os.remove(f) for f in glob.glob(config.DEST_PATH + '/*.tif') if f.endswith('_temp.tif')]

# Get Patches 






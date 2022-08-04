''' Should include all the necessary info to run the model'''

from scripts import get_patches
from scripts import get_GT

# get ground truth

# Define the paths of the images, mask layers and grid
# MASKS_PATH = 'data/geopackages/UPDATED_6_mask_per_year.gpkg'
# GRID_PATH = 'data/geopackages/grid.gpkg'
# DEST_PATH = 'data/masks'

# gt = get_GT.Get_Ground_Truth(GRID_PATH, MASKS_PATH, DEST_PATH)
# print(gt.get_items())

# # Delete the temp files 
# [os.remove(f) for f in glob.glob(DEST_PATH + '/*.tif') if f.endswith('_temp.tif') ]

# get patches 


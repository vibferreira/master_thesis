import utis
import dataset
import config
import cv2
import geopandas as gpd
import pandas as pd

##### Custom Data Split 1.0 ##### 
''' It was used to filter fine and coarser patches'''
# Read file 
geo_df = gpd.read_file(config.FILTER_PATH) # contains the idxs with a selection of non-noisy and noisy data

# Define the filters 
filters = {
       'non_veg_idxs' : geo_df.query("status == 0")['index'],
       'veg_idxs' :  geo_df.query("status == 1")['index'], 
       'mixed': geo_df.query("status == 2")['index'], 
       'single_trees_idx' : geo_df.query("status == 3")['index'], 
       'hedgerows' : geo_df.query("status == 4")['index'], 
       #'coarse_to_very_coarse': geo_df.query("status == 5")['index'][5:]
          }

# fix bug of the test_size
data_portion = 'all_coarse_labels' # ['all_coarse_labels','coarse_plus_fine_labels', 'fine_labels', 'coarse_labels']

n_patches = 283
X_train, y_train, X_val, y_val, X_test, y_test = utis.custom_split(filters, test_size=40, 
                                                                   image_paths=config.image_paths, 
                                                                   mask_paths=config.mask_paths,  
                                                                   data_portion=data_portion,
                                                                   DEST_PATH = config.TEST_DATASET_PATH,
                                                                   number_training_patchs=n_patches)
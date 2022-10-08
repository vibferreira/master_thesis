import utis
import dataset
import config
import cv2
import geopandas as gpd
import pandas as pd
import random

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from dataset import HistoricalImagesDataset


##### Define transformations #####
val_transform = A.Compose(
    [A.Normalize(mean=(0.512), std=(0.167)),
    ToTensorV2()])

train_transform = A.Compose([
      A.Rotate(limit=40,p=0.9, border_mode=cv2.BORDER_CONSTANT), # p stands for the probability with which the transformations are applied
      A.HorizontalFlip(p=0.9),
      A.VerticalFlip(p=0.9), 
      A.Transpose(p=0.9),
      A.CLAHE(p=1),
      A.Normalize(mean=(0.512), std=(0.167)),
      ToTensorV2()])

##### Dataloader ##### 
def loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    # Datasets
    train_dataset = HistoricalImagesDataset(X_train, y_train, transform=train_transform, split_type=None)
    val_dataset = HistoricalImagesDataset(X_val, y_val, transform=val_transform, split_type=None)
    test_dataset = HistoricalImagesDataset(X_test, y_test, transform=val_transform, split_type='test')

    data = next(iter(train_dataset))
    print('shape train image', data[0].shape, 'shape train mask', data[1].shape) 

    # Dataloader
    print("Training set size: ", len(train_dataset))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size = config.BATCH_SIZE, shuffle=True)
    print("Validation set size: ", len(val_dataset))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size = config.BATCH_SIZE, shuffle=True)

    print("Testing set size: ", len(test_dataset))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size = 1)
    
    return train_dataloader, val_dataloader

##### Custom Data Split ##### 
if config.split == 'tile': 
    print('Performing split based on geo tiles')
    data_split = utis.split_by_tile(config.FILTER_PATH, config.TILE_PATH, config.image_paths, config.mask_paths)
    X_test, X_train, X_val = data_split['img']['test'], data_split['img']['train'], data_split['img']['val']
    y_test, y_train, y_val = data_split['msk']['test'], data_split['msk']['train'], data_split['msk']['val']
    
    # Dataloaders
    train_dataloader, val_dataloader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)
    
elif config.split == 'filter':
    print('Performing split based on filters')
    
    # Read file 
    geo_df = gpd.read_file(config.FILTER_PATH) # contains the idxs with a selection of non-noisy and noisy data

    #Define the filters 
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
    # Dataloaders
    train_dataloader, val_dataloader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)
    
elif config.split == 'number_of_patches':
    print('Performing split based on number of patches')
    data_split = utis.split_by_tile(config.FILTER_PATH, config.TILE_PATH, config.image_paths, config.mask_paths)
    X_test, X_train, X_val = data_split['img']['test'], data_split['img']['train'], data_split['img']['val']
    y_test, y_train, y_val = data_split['msk']['test'], data_split['msk']['train'], data_split['msk']['val']
    
    # total number of idxs that are not in the X_test
    val_train_idxs = utis.train_images_paths([utis.get_file_index(i) for i in config.image_paths], [utis.get_file_index(i) for i in X_test])
    
##### Dataloader ##### 

# # Datasets
# train_dataset = HistoricalImagesDataset(X_train, y_train, transform=train_transform, split_type=None)
# val_dataset = HistoricalImagesDataset(X_val, y_val, transform=val_transform, split_type=None)
# test_dataset = HistoricalImagesDataset(X_test, y_test, transform=val_transform, split_type='test')

# data = next(iter(train_dataset))
# print('shape train image', data[0].shape, 'shape train mask', data[1].shape) 

# # Dataloader
# print("Training set size: ", len(train_dataset))
# train_dataloader = DataLoader(dataset=train_dataset, batch_size = config.BATCH_SIZE, shuffle=True)
# print("Validation set size: ", len(val_dataset))
# val_dataloader = DataLoader(dataset=val_dataset, batch_size = config.BATCH_SIZE, shuffle=True)

# print("Testing set size: ", len(test_dataset))
# test_dataloader = DataLoader(dataset=test_dataset, batch_size = 1)


# sampled patches idxs
def dataloading(number_training_patches):
    random.seed(42)
    sampled_all_idxs = random.sample(val_train_idxs, number_training_patches)

    # get paths based on the idxs
    X_paths = utis.filtered_paths(config.image_paths, sampled_all_idxs) 
    y_paths = utis.filtered_paths(config.mask_paths, sampled_all_idxs)

    # split into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_paths, y_paths, test_size=0.2, random_state=0, shuffle=True)

    # Datasets
    train_dataset = HistoricalImagesDataset(X_train, y_train, transform=train_transform, split_type=None)
    val_dataset = HistoricalImagesDataset(X_val, y_val, transform=val_transform, split_type=None)
    test_dataset = HistoricalImagesDataset(X_test, y_test, transform=val_transform, split_type='test')

    # Dataloader
    print("Training set size: ", len(train_dataset))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size = config.BATCH_SIZE, shuffle=True)
    print("Validation set size: ", len(val_dataset))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size = config.BATCH_SIZE, shuffle=True)

    print("Testing set size: ", len(test_dataset))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size = 1)

    return train_dataloader, val_dataloader
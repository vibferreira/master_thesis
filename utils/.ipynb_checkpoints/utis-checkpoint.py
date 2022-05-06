'''Defines utils functions to save, plot and make predictions '''
''' Functions to plot using torchvision make_grid are from https://medium.com/analytics-vidhya/do-you-visualize-dataloaders-for-deep-neural-networks-7840ae58fee7'''

import glob
import os
import re
import random
import numpy as np
from matplotlib import pyplot as plt, cm
from torchvision.utils import make_grid
import segmentation_models_pytorch
from sklearn.model_selection import train_test_split

import rasterio as rio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from tqdm import tqdm
import cv2

import model
import metrics
import config
import utis

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb


import torch

def plot_comparison(x:torch.Tensor, 
                    pred:torch.Tensor, 
                    y:torch.Tensor) -> None:
    
    img = np.squeeze(x.data.cpu().cpu().numpy())
    pred = np.squeeze(pred.data.cpu().cpu().numpy())
    gt = np.squeeze(y.data.cpu().cpu().numpy())

    _, ax = plt.subplots(1, 3, sharey='row')
    
    plt.figure()
    cmap = cm.get_cmap('gray') # define color map
    plt.gray()
    ax[0].imshow(img)
    ax[0].set_title('Image')

    ax[1].imshow(gt, cmap=cmap, vmin=0) # 0 are balck and white are 1  
    ax[1].set_title('Ground Truth')

    ax[2].imshow(pred, cmap=cmap, vmin=0)
    ax[2].set_title(f'Prediction')
    plt.show()
    
def save_best_model(model, 
                    dest_path: str, 
                    val_dic:dict, 
                    e:int,
                    data_portion: str,
                    rate_of_coarse_labels:float = None) -> None:
    
    ''' Saves the best model
    Args: 
    model (class): instance of the the model class
    dest_path (str): destination path
    val_dict (dict): dictionary storing valdation accuracies
    e (int): epoch, 
    data_portion (str): type of the data used ['coarse_plus_fine_labels', 'fine_labels', 'coarse_labels']'''
    
    iou = float(val_dic['IoU_val'][-1])
    acc = float(val_dic['val_accuracy'][-1])
    # [os.remove(f) for f in glob.glob(dest_path + '/*') if f.startswith(data_portion, 14)] # remove previous saved files 
    
    if data_portion == 'all_coarse_labels':
        # remove previous saved files 
        [os.remove(f) for f in glob.glob(dest_path + '/coarse_sizes' + '/*') if f.startswith(f'/rate_{rate_of_coarse_labels}_{data_portion}', 26)] 
        return torch.save(model.state_dict(), dest_path + '/coarse_sizes' + f'/rate_{rate_of_coarse_labels}_{data_portion}_best_model_epoch_{e+1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')
    
    else:
        # remove previous saved files
        [os.remove(f) for f in glob.glob(dest_path + '/*') if f.startswith(data_portion, 14)] 
        return torch.save(model.state_dict(), dest_path + f'/{data_portion}_best_model_epoch_{e+1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')


def plot_grids(grids, titles = ["Input", 'Target']):
    nrow = len(grids)
    fig = plt.figure(figsize=(8, nrow), dpi=300)
    # Remove the space between the grids
    fig.subplots_adjust(wspace=0, hspace=0)
    # Each grid is new subplot
    for i in range(1,nrow+1):
        sub = fig.add_subplot(nrow,1,i)
        # Dont show the x-axis
        sub.xaxis.set_visible(False)
        # Remove y axis ticks and set yaxis label
        sub.set_yticks([])
        sub.set_ylabel(titles[i-1], rotation=0, fontsize=5, labelpad=15)
        sub.imshow(grids[i-1])
    plt.show()

def create_grids(imgList, nrofItems, pad, norm = True):
    '''Create a list of Image grids from Torch tensors'''
    # Permute the axis as numpy expects image of shape (H x W x C) 
    return list(map(lambda imgs: make_grid(imgs[:nrofItems], normalize=norm, padding = pad, nrow=nrofItems).permute(1, 2, 0), imgList))

# Create list of two image grids - Input Images & Target Mask Images for Segmentation task
def create_segement_grids(loader_iter, nrofItems = 5, pad = 4):
    '''# Create list of two image grids - Input Images & Target Mask Images for Segmentation task'''
    inp, target = next(loader_iter)
    return create_grids([inp, target], nrofItems, pad)
    
# def remove_paths_with_more_than_one_class(mask_paths:list, image_paths:list) -> list:
#     '''Returns only images that does not contain more than one label'''
#     i = 0
#     for mask, img in zip(mask_paths, image_paths):
#         data = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
#         if len(np.unique(data)) > 1:
#             image_paths.remove(img)
#             mask_paths.remove(mask)
#             i+=1
#     print(f'{i} paths removed')
#     return mask_paths, image_paths

def train_images_paths(paths:list, test:list) -> list:
    ''' Returns list of images paths that DOES NOT exist on the test list'''
    train_paths = []
    for i in paths:
        if not np.isin(i, test):
            train_paths.append(i)
    return train_paths

def get_file_index(file:str) -> str:
    '''Returns the idx name from file name'''
    return re.split(r"[/_.]\s*", file)[-2]

def filtered_paths(current_paths:list, 
                   filter_paths:list)-> list:
    ''' Returns only the paths that match the filter index and does not conain more than one label'''
    filtered_paths = []
    for i in current_paths:
        if np.isin(int(get_file_index(i)), list(filter_paths)):
            filtered_paths.append(i)      
    return filtered_paths

def custom_split(filters:dict, image_paths:list, 
                 mask_paths:list, 
                 test_size:int, 
                 data_portion:str, 
                 rate_of_coarse_labels = 1) -> list:
    
    ''' Split the dataset based on the filter status
    Args:
    filters (dict): dict with the filtered paths by status
    test_size (float): percentage of the dataset that will be used to test
    data_portion (str):  'coarse_plus_fine_labels', 'fine_labels' or 'coarse_labels' define which portion of the data to use'''

    # sample to the same size of the smallest dataset
    veg_filters = filters.copy()
    
    # coarse to very coarse is not considered in the fine dataset filter 
    if np.isin('coarse_to_very_coarse', list(veg_filters.keys())):
        del veg_filters['coarse_to_very_coarse']
    
    # find the minimum dataset size among the classs in the filter, so all of them are equally represented    
    sample_size = min([len(value) for key, value in veg_filters.items()])
   
    # random sample sample_size points from each filter class
    new_dic = {key:value.sample(n=sample_size, replace=False, random_state=0) for key, value in veg_filters.items()} # random sample

    # sample from each category the sample number of samples, lets say 8, so the final dataset has is balanced 
    test_size = test_size/(sample_size*5) # 5 is the number of filters
    test_size = int(sample_size*test_size)
    test_idx = np.concatenate([value.sample(n=test_size, replace=False, random_state=42) for value in new_dic.values()])
    val_idx = np.concatenate([value.sample(n=test_size, replace=False, random_state=0) for value in new_dic.values()])

    # train consists of the remaining dataset
    t_train = np.concatenate([value for value in new_dic.values()])
    train_idx = train_images_paths(t_train, list(test_idx) + list(val_idx)) # all dataset minus test and val

    # Filter the paths from the path lists
    # Note the test dataset is the SAME for all data portions
    X_test = filtered_paths(image_paths, test_idx) 
    y_test = filtered_paths(mask_paths, test_idx) 
    
    portions = ['all_coarse_labels','coarse_plus_fine_labels', 'fine_labels', 'coarse_labels']
    assert np.isin(data_portion, portions)
    
    # Decide if using whole data or ONLY the filtered paths 
    if data_portion == 'coarse_plus_fine_labels': # all patches are used
        X_train = train_images_paths(image_paths, X_test)
        y_train = train_images_paths(mask_paths, y_test)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=True)
        
    elif data_portion == 'all_coarse_labels': # all patches but the fine ones
        # Get a list with all fine idxs 
        idxs = list(test_idx) + list(val_idx) + list(train_idx)  # all fine idx               
        fine_X_idxs = filtered_paths(image_paths, idxs) 
        fine_y_idxs = filtered_paths(mask_paths, idxs)  
          
        # List of ALL coarse labels 
        X_train = train_images_paths(image_paths, fine_X_idxs) # all patches but the fine ones
        all_coarse_idxs = [get_file_index(i) for i in X_train]
        
        # select random percentage of coarse labels
        random.seed(41)
        sampled_coarse_idxs = random.sample(all_coarse_idxs, int(len(all_coarse_idxs)*rate_of_coarse_labels))
        
        # get the paths 
        coarse_X_idxs = filtered_paths(image_paths, sampled_coarse_idxs) 
        coarse_y_idxs = filtered_paths(mask_paths, sampled_coarse_idxs)
        
        X_train, X_val, y_train, y_val = train_test_split(coarse_X_idxs, coarse_y_idxs, test_size=0.20, random_state=42, shuffle=True)
        
    elif data_portion == 'coarse_labels': # if only coarse patches are used
        X_train = filtered_paths(image_paths, filters['coarse_to_very_coarse'])
        y_train = filtered_paths(mask_paths, filters['coarse_to_very_coarse'])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=40, random_state=42, shuffle=True)     
        
    elif data_portion == 'fine_labels': # if only the fine patches are used 
        X_train = filtered_paths(image_paths, train_idx) 
        y_train = filtered_paths(mask_paths, train_idx) 

        X_val = filtered_paths(image_paths, val_idx) 
        y_val = filtered_paths(mask_paths, val_idx) 

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_coords (image_list:list) -> dict:
    '''
    Return the rasterio profile of the images
    Args:
    image_list (list) : list of image paths
    Returns:
    coords (dict) : rasterio profile with image id
    '''
    coords = {}
    for img in image_list:
        # and its respective id
        ids = get_file_index(img)

        # get the profile of the patches
        with rio.open(img) as src:
            ras_data = src.read().astype('uint8')
            profile = src.profile # get the original image profile

        # save in a dict
        coords.update([(ids, profile)])
    return coords

def custom_save_patches(patch: torch.Tensor,
                        coords: dict, 
                        file_id: str,
                        batch_idx: int,
                        folder: str,
                        subfolder:str = 'masks') -> None:
    '''
    Saves the patches according to the original crs
    
    Args:
    patchs(torch.Tensor): patches to save
    coords(dict): rasterio profile with image id
    file_id(str): patch id
    batch_idx(int): batch number
    folder(str): path to folder
    subfolder(str): subfolder name
    
    Returns:
    None
    '''
    # torch to numpy
    patch = np.squeeze(patch.detach().cpu().numpy())
    
    file_path = f"{folder}/{subfolder}/{subfolder}_{batch_idx}_id_{file_id}.png"

    transform = from_origin(coords[file_id]['transform'][2],coords[file_id]['transform'][5],0.75,0.75)
    crs = CRS.from_epsg('2154')

    with rio.open(file_path, 'w',
                driver='GTiff',
                height=patch.shape[0],
                width=patch.shape[1],
                dtype=patch.dtype,
                count=1, # number of bands, CAREFUL if the image has RGB
                crs = crs, 
                transform=transform) as dst:
                dst.write(patch[np.newaxis,:,:]) # add a new axis, required by rasterio 
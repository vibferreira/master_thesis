''' Defines utils functions '''
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
from pathlib import Path
import pandas as pd

import rasterio as rio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from tqdm import tqdm
import cv2

import model
import metrics
import config
import utis
import train_val_test

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'iframe'
import plotly.express as px

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
    
    if data_portion == 'coarse_labels':
        # remove previous saved files 
        [os.remove(f) for f in glob.glob(dest_path + '/coarse_sizes' + '/*') if f.startswith(f'/rate_{rate_of_coarse_labels}_{data_portion}', 26)] 
        return torch.save(model.state_dict(), dest_path + '/coarse_sizes' + f'/rate_{rate_of_coarse_labels}_{data_portion}_best_model_epoch_{e+1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')
    
    elif data_portion == 'fine_labels':
        [os.remove(f) for f in glob.glob(dest_path + '/fine_sizes' + '/*') if f.startswith(f'/rate_{rate_of_coarse_labels}_{data_portion}', 24)] 
        return torch.save(model.state_dict(), dest_path + '/fine_sizes' + f'/rate_{rate_of_coarse_labels}_{data_portion}_best_model_epoch_{e+1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')
    
    else:
        # remove previous saved files
        [os.remove(f) for f in glob.glob(dest_path + '/*') if f.startswith(data_portion, 14)] 
        return torch.save(model.state_dict(), dest_path + f'/{data_portion}_best_model_epoch_{e+1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')

    
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

def get_n_training_samples(image_paths,
                           mask_paths,
                           filtered_idxs, 
                           number_training_patchs):
    
        all_coarse_idxs = [get_file_index(i) for i in filtered_idxs]

        # select random percentage of coarse labels
        random.seed(42)
        val_samples = int(0.30*number_training_patchs)
        n_samples = number_training_patchs + int(0.30*number_training_patchs)
        sampled_coarse_idxs = random.sample(all_coarse_idxs, n_samples)
        
        # get the paths 
        X_idxs = filtered_paths(image_paths, sampled_coarse_idxs) 
        y_idxs = filtered_paths(mask_paths, sampled_coarse_idxs)
        
        return val_samples, X_idxs, y_idxs
    
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

def save_test_dataset(DEST_PATH:str,
                      list_of_imgs:list) -> None:
    
    ''' Save the test dataset in a separate folder
    Args:
    DEST_PATH (str): destination folder
    list_of_imgs(list): list of images to save'''
    
    coords = get_coords(list_of_imgs)
    
    if not glob.glob(DEST_PATH +'/*.tif'):
        #save test dataset on the dest folder WITH CRS
        for img in list_of_imgs:
            idx = utis.get_file_index(img)
            print(img)
            with rio.open(img) as src:
                data = src.read().astype('uint8')
                profile = src.profile

            print(f'Saving dataset {DEST_PATH}/1942_{idx}.tif')
            with rio.open(f'{DEST_PATH}/1942_{idx}.tif', 'w', **profile) as dst:
                dst.write(np.squeeze(data), 1)  
            
def custom_split(filters:dict, image_paths:list, 
                 mask_paths:list, 
                 test_size:int, 
                 data_portion:str, 
                 DEST_PATH:str, 
                 number_training_patchs = 20) -> list:
    
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

    # get all veg_filter idx
    all_filter_idx = np.concatenate([i for i in veg_filters.values()])
    
    # random sample sample_size points from each filter class to create the TEST DATASET
    test_size = test_size/(sample_size*5) # 5 is the number of filters
    test_size = int(sample_size*test_size)
    test_dic = {key:value.sample(n=test_size, replace=False, random_state=0) for key, value in veg_filters.items()} # random sample
    test_idx = np.concatenate([i for i in test_dic.values()])
    
    # get the remaining patches from the filters to be the X_train and X_val of the fine patches
    random.seed(42)
    val_train_idxs = train_images_paths(all_filter_idx, test_idx)
    if not data_portion == 'fine_patches_WITH_X_test':
        sampled_all_idxs = random.sample(val_train_idxs, number_training_patchs)

    # Filter the paths from the path lists
    # Note the test dataset is the SAME for all data portions
    X_test = filtered_paths(image_paths, test_idx) 
    y_test = filtered_paths(mask_paths, test_idx) 
    
#     # save a separate test dataset 
    if data_portion == 'fine_labels':
        save_test_dataset(f'{DEST_PATH}/images', X_test)
        save_test_dataset(f'{DEST_PATH}/masks', y_test)
        
    X_test = glob.glob(f'{DEST_PATH}/images' +'/*.tif') 
    y_test = glob.glob(f'{DEST_PATH}/masks' +'/*.tif') 
    
    portions = ['fine_patches_but_X_test', 'fine_patches_WITH_X_test', 'all_coarse_labels','fine_labels', 'coarse_labels', 'all_coarse_labels']
    assert np.isin(data_portion, portions)
    
    # Decide if using whole data or ONLY the filtered paths 
    if data_portion == 'fine_patches_but_X_test': 
        # get the paths 
        # all patches BUT the X_test
        X_idxs = filtered_paths(image_paths, sampled_all_idxs) 
        y_idxs = filtered_paths(mask_paths, sampled_all_idxs)
        
        return X_idxs, y_idxs
    
    if data_portion == 'fine_patches_WITH_X_test': 
        # get the paths 
        # all patches BUT the X_test
        print('ATTENTION all filter idx', len(all_filter_idx))
        X_idxs = filtered_paths(image_paths, all_filter_idx) 
        y_idxs = filtered_paths(mask_paths, all_filter_idx)
        return X_idxs, y_idxs
    
    elif data_portion == 'all_coarse_labels':
        X_test = filtered_paths(image_paths, all_filter_idx) # USE ALL FINE PATHS FOR TESTING
        y_test = filtered_paths(mask_paths, all_filter_idx) # USE ALL FINE PATHS FOR TESTING
        
        X_train = train_images_paths(image_paths, X_test)
        y_train = train_images_paths(mask_paths, y_test)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, shuffle=True)   
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    if data_portion == 'coarse_labels': # if only coarse patches are used
        coarse_X_idx = filtered_paths(image_paths, val_train_idxs) 
        coarse_y_idx = filtered_paths(mask_paths, val_train_idxs) 
        
        val_samples, coarse_X_idx, coarse_y_idx = get_n_training_samples(image_paths,
                           mask_paths,
                           coarse_X_idx, 
                           number_training_patchs)

        X_train, X_val, y_train, y_val = train_test_split(coarse_X_idx, coarse_y_idx, test_size=val_samples, random_state=42, shuffle=True)    
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    elif data_portion == 'fine_labels': # if only the fine patches are used 
        fine_X_idx = filtered_paths(image_paths, val_train_idxs) 
        fine_y_idx = filtered_paths(mask_paths, val_train_idxs) 
        
        val_samples, fine_X_idx, fine_y_idx = get_n_training_samples(image_paths,
                           mask_paths,
                           fine_X_idx, 
                           number_training_patchs)

        X_train, X_val, y_train, y_val = train_test_split(fine_X_idx, fine_y_idx, test_size=val_samples, random_state=29, shuffle=True) 

        return X_train, y_train, X_val, y_val, X_test, y_test

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
    
    file_path = f"{folder}/{subfolder}/{subfolder}_{batch_idx}_id_{file_id}.tif"

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

def count_data_dist(y_test: list) -> np.array :
    ''' Count bins on numpy array
    y_test(list): list of mask images'''
    counts = np.zeros((len(y_test),2))

    for i, img in enumerate(y_test):
        # read image
        IMG = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # count 0 and 1
        x = np.bincount(IMG.reshape(-1))
        # set 0 to label one when only black imgs is available
        if len(x) < 2:
            x = [x[0], 0]

        # store in an array    
        counts[i] = x

    return counts.sum(axis=0)

def plot_pizza(y_test: list, title:str) -> None:
    ''' Plot pizza graph of the data distribution
    y_test(list): list of mask images'''
        
    f_counts = count_data_dist(y_test)
    df_avg_veg = {'average vegetation count': f_counts, 'label': ['non-trees','trees']}

    fig = px.pie(df_avg_veg, names='label', 
                 values='average vegetation count', 
                 color='label',
                 color_discrete_map={'non-trees':'#B9948C','trees':'#2E8B57'}, 
                 title=f'Percentage of trees and non-trees in the {title}')

    fig.update_layout(legend=dict(
        yanchor="top",
        y= 0.8,
        xanchor="right",
        x = 1.1, 
        font=dict(size=15),    
        ))
    fig.show(renderer="svg")
    
def iou_on_test_dataset(MODEL:str, 
                        number_patches:list,
                        dataloader,
                        X_test,
                        folder) -> dict:
    ''' Return dict with the IoU according to number 
    of patches the model was trained with
    Args:
    MODEL(str): pytorch model saved with the weights already sorted by number of patches
    number_patches(list): list with the number of patches already sorted '''
    
    iou_all = {}
    for i, rate in zip(MODEL, number_patches):
        # load the model 
        model_ = model.unet_model.to(config.DEVICE)
        model_.load_state_dict(torch.load(i))

        # get the y_hat and y_true
        y_hat, y_true, y_score = train_val_test.make_predictions(model_, dataloader, X_test, folder, print_pred=False, save_patches=False)

        # get IoU
        all_metrics = metrics.metrics(y_hat, y_true)
        iou = all_metrics['iou']

        # save dict
        iou_all.update([(rate,iou)])
    return iou_all

def create_new_dir(new_dir_path:str) -> None:
    ''' create a new dir
    Args:
    new_dir_path(str): new directory path'''
    os.makedirs(new_dir_path)
    
def save_model(model_to_save, 
               dir_to_create, 
               fold, 
               dic_results, 
               epoch, 
               path_to_save_models) -> None:
    
    # path name
    path_save = f'{dir_to_create}/fold_{fold}_epoch_{epoch}_iou_{dic_results:.3f}.pth'
    
    # remove previous files saved with the same fold name
    path_list = glob.glob(dir_to_create + '/*')
    [os.remove(f) for f in path_list if f'fold_{fold}' in f] #27 or 26
    
    # save the model
    torch.save(model_to_save.state_dict(), path_save)

def mean_per_folder(MODELS: list) -> dict:
    ''' Calculate the mean per folder
    Args:
    MODELS (list): list of models
    Return:
    dict: dictionaty with folder name as key and mean iou as value'''
    #
    mean_kfold_iou = np.zeros(len(MODELS))
    std_dev_iou = np.zeros(len(MODELS))
    dic = {}
    # file path

    n_patches = re.split(r"[/_]\s*", MODELS[0])[5]
    # store the iou per fold 
    for i, model in enumerate(MODELS): 
        iou = float(re.split(r"[/_]\s*", model)[-1][:-4])
        mean_kfold_iou[i] = iou 
    # calculate the mean
    dic[n_patches] = [round(mean_kfold_iou.mean(),3), round(mean_kfold_iou.std(),3)]
    
    
    return dic

def get_mean_independet_test_dataset(MODELS:list, 
                                     test_dataloader, 
                                     X_test:list):
    
    n_patches = re.split(r"[/_]\s*", MODELS[0])[5]
    iou_mean = np.zeros(len(MODELS))
    dic = {}
    
    for n, i in enumerate(MODELS):
        model_ = model.unet_model.to(config.DEVICE)
        model_.load_state_dict(torch.load(i))

        # get the y_hat and y_true
        y_hat, y_true, y_score = train_val_test.make_predictions(model_, 
                                                                 test_dataloader, 
                                                                 X_test, 
                                                                 folder=None, 
                                                                 print_pred=False, 
                                                                 save_patches=False)
        # get IoU
        all_metrics = metrics.metrics(y_hat, y_true)
        iou = all_metrics['iou']
        iou_mean[n] = iou 
    
    dic[n_patches] = [iou_mean.mean(), iou_mean.std()]
    
    return dic

def interact_over_folder_mean(file_path, 
                              independent_test=False, 
                              test_dataloader=None, 
                              X_test=None) -> dict: 
    # list the subdirectories 
    list_subdirectories = [f for f in file_path.iterdir() if f.is_dir() and '.ipynb_checkpoints' not in str(f)]
    final_means= {}
    
    for sub_dic in list_subdirectories:
        # list the files in each subdirectory 
        list_files_each_folder = list(Path(sub_dic).glob('**/*'))
        list_files_each_folder.sort()
        list_files_each_folder =  [str(d) for d in list_files_each_folder if '.ipynb_checkpoints' not in str(d)]
        
        # get the simple mean
        dic = mean_per_folder(list_files_each_folder)
        
        # get the mean in the independent dataset
        if independent_test:
            dic = get_mean_independet_test_dataset(list_files_each_folder, test_dataloader, X_test)
            
        else:
            # get the simple mean
            dic = mean_per_folder(list_files_each_folder)
        
        final_means.update(dic)
        
    return final_means

def get_DF_with_the_means(my_file, 
                          label,
                          independent_test=False, 
                          test_dataloader=None, 
                          X_test=None) -> pd.DataFrame: 
    
    # GET DICT WITH THE MEANS
    if independent_test:
        means = interact_over_folder_mean(my_file, independent_test=True, test_dataloader=test_dataloader, X_test=X_test)
    else:
        means = interact_over_folder_mean(my_file) # 
        
    df = pd.DataFrame.from_dict(means, orient='index').reset_index()
    df.columns = ['N_PATCHES', 'IOU', 'STD_DEV']
    
    # CREATING A COLUMN WITH THE FILE NAME
    file_name=re.split(r"[/_.]\s*", str(my_file))[-2]
    df = df.assign(DATA_PORTION = lambda x: f'{file_name}_{label}')
    
    # CHANGE TO NUMERIC DATA TYPE
    df['N_PATCHES'] = pd.to_numeric(df['N_PATCHES']) 
    df.sort_values(by = 'N_PATCHES', inplace=True) # SORTING BY NUMBER OF PATCHES
    
    return df 

def bar_chat_datasets(masks:list, 
                      names:list, 
                      x_axis:str) -> px.bar:
    
    ''' 
    Plot bar chart comparing the percentage of non-trees (0) and trees (1) in each dataset (train, val, test)
    Args:
    masks(list): list of datasets
    names(list) : list of datasets names
    '''

    df_avg_veg ={}
    for mask, name in zip(masks, names): 
        f_counts = count_data_dist(mask)
        df_avg_veg.update({name: f_counts})

    df = pd.DataFrame(df_avg_veg)
    df = df.melt() # invert axis 
    df['label'] = ['non-trees','trees', 'non-trees','trees', 'non-trees','trees'] # name accordingly 
    df['rate'] = df.groupby('variable').transform(lambda x: (x/x.sum())*100) # calculate the percentage
    df['rate'] = df['rate'].apply(lambda x: round(x,3)) # calculate the percentage
    
    # Plotting it 
    fig = px.bar(df, x="variable", y="rate",
             color='label', barmode='group',
             height=400,
             template="plotly_white",
             text_auto=True,
             color_discrete_map={
                "non-trees": "#99d8c9",
                "trees": "#2ca25f"})

    fig.update_layout(
        title = 'Percentage of non-trees and trees in each dataset',
        xaxis_title=f"{x_axis} Dataset",
        yaxis_title="Rate",
        legend_title="",
        font=dict(size=13))

    fig.show()
    
def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands.
    !!!!!!!!!! CODE FROM STACKOVERFLOW'S user171780: https://stackoverflow.com/questions/61494278/plotly-how-to-make-a-figure-with-multiple-lines-and-shaded-area-for-standard-de !!!!!!!!!
    """
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = True,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
        # fig.update_traces(name=f'Std_Deviation',showlegend = True)
        fig.data[0].name = f'STD DEV Fine Model'
        fig.data[2].name = f'STD DEV Coarse Model'
        fig.update_layout(legend_traceorder="reversed+grouped")
    return fig
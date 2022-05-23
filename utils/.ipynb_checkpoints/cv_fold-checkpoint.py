import glob
import copy
import os
import numpy as np

import model
import metrics
import config
import utis
import train_val_test
from dataset import HistoricalImagesDataset

import cv2
import geopandas as gpd
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


geo_df = gpd.read_file(config.FILTER_PATH) # contains the idxs with a selection of non-noisy and noisy data

def kfold_cross_validation(n_splits, 
                           filters, 
                           val_transform, 
                           train_transform, 
                           save_path,
                           scaler) -> None: 
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    save_models = {}

    for n_patches in [370]: #np.arange(20, 370, 35)

        # Define X_train, X_val, X_test
        data_portion = 'all_labels'
        X_train, y_train = utis.custom_split(filters, test_size=40, 
                                                                   image_paths=config.image_paths, 
                                                                   mask_paths=config.mask_paths,  
                                                                   data_portion='all_labels',
                                                                   DEST_PATH = config.TEST_DATASET_PATH,
                                                                   number_training_patchs=n_patches)
        # Datasets
        train_dataset = HistoricalImagesDataset(X_train, y_train, transform=train_transform, split_type=None)

        # load the datasetsuscuic
        # datasets = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        results = {}

        dir_to_create = os.path.join(save_path, str(n_patches))
        print(dir_to_create)

        # # initialize a dictionary to store TRAINING history (keep track on training)
        training_history = {"avg_loss": [], "accuracy": [], "iou":[]} # initialize a dictionary to store TRAINING history (keep track on training)

        # # # initialize a dictionary to store VALIDATION history (keep track on VALIDATION)
        validation_history = {"avg_val_loss": [], "val_accuracy": [], "IoU_val":[]}

        for fold,(train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
            print('------------fold no---------{}----------------------'.format(fold))
            print(f'''Training the network for {config.NUM_EPOCHS} epochs, with a batch size of {config.BATCH_SIZE}''') # try with logger

            # best iou accuracy to save
            best_accuracy = 0.0

            # get the dataloader based on the kfold indexs
            trainloader, testloader = train_val_test.k_fold_dataloaders(train_idx, 
                            test_idx, 
                            train_dataset)

            # Model, optmizer, loss
            unet = model.unet_model.to(config.DEVICE) # initialize the model
            opt = optim.Adam(unet.parameters(), lr=config.LR)
            lossFunc = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            scaler = GradScaler()

            # reset weights
            # unet.apply(train_val_test.reset_weights)

            for epoch in range(0, config.NUM_EPOCHS):
                trained = train_val_test.training(network=unet, 
                                                  trainloader=trainloader, 
                                                  optimizer=opt, loss_function=lossFunc, 
                                                  save_path=save_path, epoch=epoch, training_history=training_history, scaler=scaler)
                dic_results, validated = train_val_test.val(unet, testloader, epoch, lossFunc, validation_history, fold)

                # create a folder named with the number of patches used o train
                if not os.path.exists(dir_to_create):
                    utis.create_new_dir(dir_to_create)

                # Save best model
                if dic_results > best_accuracy and epoch > 10: # maybe add a minimum number of epochs as conditions
                    # Saving the model
                    results[fold] = dic_results
                    utis.save_model(unet, dir_to_create, fold, dic_results,epoch)
                    best_accuracy = dic_results
    

            save_models[fold] = copy.deepcopy(unet)
            del unet
            torch.cuda.empty_cache()

# Testing if it works 
if __name__ == '__main__':
    # Define the filters 
    filters = {'non_veg_idxs' : geo_df.query("status == 0")['index'],
           'veg_idxs' :  geo_df.query("status == 1")['index'], 
           'mixed': geo_df.query("status == 2")['index'], 
           'single_trees_idx' : geo_df.query("status == 3")['index'], 
           'hedgerows' : geo_df.query("status == 4")['index']}

    # Define transformations
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
    
    # Autocasting 
    scaler = GradScaler()
    
    # calling cross validation
    kfold_cross_validation(5, 
                           filters, 
                           val_transform, 
                           train_transform, 
                           save_path = 'best_model/fine_sizes', 
                           scaler=scaler)

    # one_path = 'best_model/coarse_sizes/20'
    # print(glob.glob(one_path + '/*'))

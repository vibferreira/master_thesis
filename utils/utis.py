'''Defines utils functions to save and plot  '''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

def plot_comparison(pred, y):
    gt = np.squeeze(y.data.cpu().numpy()[0])
    pred = np.squeeze(pred.sigmoid().numpy())
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(gt)
    ax[0].set_title('Ground Truth')
    ax[1].imshow(pred[0])
    ax[1].set_title('Prediction')
    plt.show()

def save_best_model(model, 
                    dest_path: str, 
                    val_dic:dict, 
                    e) -> None:
    
    ''' Saves the best model
    Args: 
    model (class): instance of the the model class
    dest_path (str): destination path
    val_dict (dict): dictionary storing valdation accuracies'''
    
    iou = float(val_dic['IoU_val'][-1])
    acc = float(val_dic['val_accuracy'][-1])
    [os.remove(f) for f in glob.glob(dest_path + '/*')] # remove previous saved files 
    return torch.save(model.state_dict(), dest_path + f'/best_model_epoch_{e + 1}_acc_{round(acc,3)}_iou_{round(iou, 3)}.pth')
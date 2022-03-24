'''Defines utils functions to save and plot  '''
''' Functions to plot using torchvision make_grid are from https://medium.com/analytics-vidhya/do-you-visualize-dataloaders-for-deep-neural-networks-7840ae58fee7'''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


import torch

def plot_comparison(x:torch.Tensor, 
                    pred:torch.Tensor, 
                    y:torch.Tensor) -> None:
    gt = np.squeeze(y.data.cpu().cpu().numpy()[0])
    pred = np.squeeze(pred.sigmoid().cpu().numpy())
    img = np.squeeze(x.data.cpu().cpu().numpy()[0])
    _, ax = plt.subplots(1, 3, sharey='row')
    
    plt.gray()
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(gt)
    ax[1].set_title('Ground Truth')
    ax[2].imshow(pred[0])
    ax[2].set_title('Prediction')
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
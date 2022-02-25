import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import random

import matplotlib.pyplot as plt

def get_file_name(paths):
    ''' Returns file_name.tif'''
    return [img.split('\\')[-1].split('.')[0] for img in paths]
    
def plot_comparison(img1, img2, title, hist=False):
    ''' Plot comparison between two plots'''
    n_col = 2
    _, axs = plt.subplots(1, n_col, figsize=(20, 10))
    
    axs[0].imshow(img1[::10,::10], cmap='gray')
    axs[0].set_title(title[0])
    if hist:
        axs[1].hist(img2, bins=256, range= (0,255), alpha=0.8,  color='green')
        axs[1].set_title(title[1])
    else: 
        axs[1].imshow(img2[::10,::10], cmap='gray')
        axs[1].set_title(title[1])
    # plt.suptitle(title,fontsize=20)
    plt.show()

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2
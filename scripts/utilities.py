import os
import pandas as pd
import torch
import random
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
import matplotlib.pyplot as plt

def get_file_name(paths: list) -> list:
    ''' Returns file_name.tif'''
    return [img.split('\\')[-1].split('.')[0] for img in paths]
    
def plot_comp(img1: np.array, img2: np.array, title: str, hist:bool=False):
    '''Plot comparison between two plots'''
    n_col = 2
    _, axs = plt.subplots(1, n_col, figsize=(20, 10))

    axs[0].imshow(img1, cmap='gray')
    axs[0].set_title(title[0])

    axs[1].imshow(img2, cmap='gray')
    axs[0].set_title(title[1])
    plt.show()


def get_patches(years:list, 
                images:dict, 
                window:tuple, 
                save:bool = False, 
                path_name:str ='images') -> np.array:

    for year in years:
        img_patches = patchify(images[year], window, step=window[0])
        if save:
            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    single_patch_img = Image.fromarray(img_patches[i,j,:,:])
                    single_patch_img.save(f"{path_name}\{year}_{i}_{j}.tif")
    return np.concatenate(img_patches)

def get_random_pos(img: torch.Tensor, window_shape: tuple) -> tuple:
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def vitoria():
    pass
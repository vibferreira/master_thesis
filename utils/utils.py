import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

# Class Dataset
# Reads the data
# Apply CLAHE
# Clip it into patches
# Apply futher transformations? 

def get_file_name(paths):
    ''' Returns file_name.tif'''
    return [img.split('\\')[-1].split('.')[0] for img in paths]
    


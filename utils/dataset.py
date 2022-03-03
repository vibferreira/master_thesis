import numpy as np
import glob

from PIL import Image 
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

class HistoricalImagesDataset(Dataset):
    def __init__(self, data_paths:list, label_paths:list) -> None:
        # List of files 
        self.data_paths = data_paths
        self.label_paths = label_paths

    def __len__(self) -> int:
        # Lenght 
        return len(self.data_paths)
                   
    def clahe_equalized(self, img: Image) -> np.array:
        ''' Apply CLAHE equalization'''
        np_img= np.asarray(img)
        clahe = cv2.createCLAHE(clipLimit = 5)
        return clahe.apply(np_img)

    def __getitem__(self, idx:int) -> torch.Tensor: 
        ''' Get individual data corresponding to the index in the data and label paths
        Returns:
        Tensor: specific data on index converted to Tensor'''
        # Image
        image = Image.open(self.data_paths[idx])
        image = self.clahe_equalized(image)
        image = ToTensor()(image) # numpy array to a normalised tensor [0 to 1]

        # Labels 
        mask = Image.open(self.label_paths[idx])
        mask = ToTensor()(mask)
  
        return image, mask

 # Test if working
IMAGES_PATH = r'data\patches\images\1942'
MASK_PATH = r'data\patches\masks\1942'

image_paths = glob.glob(IMAGES_PATH +'\*.tif')
mask_paths = glob.glob(MASK_PATH +'\*.tif')

print('Number of image patches:', len(image_paths),'\nNumber of mask patches:', len(mask_paths))

dataset = HistoricalImagesDataset(image_paths, mask_paths)
print('Len dataset', len(dataset))

data = next(iter(dataset))

print('shape image', data[0].shape, 'shape mask', data[1].shape)       

# Train, Test, Split 

train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #-- pytorch alternative to the train_test_split command line from Scikit-Learn

train_size = int(0.5 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

print(len(train_dataset), len(val_dataset), len(test_dataset))




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






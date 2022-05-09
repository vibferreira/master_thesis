import numpy as np
import glob

from PIL import Image 
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utis

class HistoricalImagesDataset(Dataset):
    def __init__(self, data_paths:list, 
                 label_paths:list, 
                 transform:bool=None, 
                 split_type:str=None) -> None:
        
        # List of files 
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.transform = transform
        self.split_type = split_type

    def __len__(self) -> int:
        # Lenght 
        return len(self.data_paths)
    
    # def clahe_equalized(self, img: Image) -> np.array: # this step should be before
    #     ''' Apply CLAHE equalization'''
    #     np_img= np.asarray(img)
    #     clahe = cv2.createCLAHE(clipLimit = 5)
    #     return clahe.apply(np_img)

    def __getitem__(self, idx:int) -> torch.Tensor: 
        ''' Get individual data corresponding to the index in the data and label paths
        Returns:
        Tensor: specific data on index converted to Tensor'''
        # Image
        img_idx = utis.get_file_index(self.data_paths[idx])
        image = cv2.imread(self.data_paths[idx], cv2.IMREAD_GRAYSCALE)
        # image = Image.open(self.data_paths[idx])
        # image = self.clahe_equalized(image)
        # image = ToTensor()(image) # numpy array to a normalised tensor [0 to 1] # 1, 

        # Labels 
        mask = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        # mask = Image.open(self.label_paths[idx])
        # mask = ToTensor()(mask)
     
        if self.transform is not None:
            # Convert PIL image to numpy array
            # image_np = np.asarray(image)
            # mask_np = np.asarray(mask)
            
            # Apply transformations
            transformed = self.transform(image=image, mask=mask)
            
            # Convert numpy array to PIL Image
            image = transformed['image'] # 1, 256, 256
            mask = transformed['mask'] # 256, 256
            # mask = mask.unsqueeze(0) # 1, 256, 256
           
        img = {}
        msk = {} 
        if self.split_type == 'test':
            img.update([(img_idx,image)])
            msk.update([(img_idx, mask)])
            # print(img)
            return img, msk
        else:
            return image, mask





from dataset import HistoricalImagesDataset
from model import DoubleConv
# from criterion import X

import glob
from torch.utils.data import DataLoader, random_split

IMAGES_PATH = r'data\patches\images\1942'
MASK_PATH = r'data\patches\masks\1942'

image_paths = glob.glob(IMAGES_PATH +'\*.tif')
mask_paths = glob.glob(MASK_PATH +'\*.tif')

print('Number of image patches:', len(image_paths),'\nNumber of mask patches:', len(mask_paths))

dataset = HistoricalImagesDataset(image_paths, mask_paths)

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

# Call the model

DoubleConv(1,2)
# Training Loop 


# Validation Loop 


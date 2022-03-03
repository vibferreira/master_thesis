from pyexpat import model
from tqdm import tqdm
import glob

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from dataset import HistoricalImagesDataset
import config
from criterion import cross_entropy_loss
import segmentation_models_pytorch as smp

# Dataset Object 
dataset = HistoricalImagesDataset(config.image_paths, config.mask_paths)
data = next(iter(dataset))
print('shape image', data[0].shape, 'shape mask', data[1].shape)       

# Train, Test, Split 
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #-- pytorch alternative to the train_test_split command line from Scikit-Learn

train_size = int(0.5 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader
print("Training set size: ", len(train_dataset))
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE)
print("Validation set size: ", len(val_dataset))
val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE)
print("Test set size: ", len(test_dataset))
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE)

model = smp.Unet(
    encoder_name= config.BACKBONE,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=config.N_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes= config.NUM_CLASSES,    # model output channels (number of classes in your dataset)
)

# print(model)


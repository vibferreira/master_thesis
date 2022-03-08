from pyexpat import model
from tqdm import tqdm
import glob
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

import model
import metrics
from dataset import HistoricalImagesDataset
import config
import criterion

import matplotlib.pyplot as plt


# Dataset Object 
print('Number of image patches:', len(config.image_paths),'\nNumber of mask patches:', len(config.mask_paths))
dataset = HistoricalImagesDataset(config.image_paths, config.mask_paths)
data = next(iter(dataset))
print('shape image', data[0].shape, 'shape mask', data[1].shape)       

# Train, Test, Split -- DEVEOLP A SPLITTING STRATEGY BASED ON THE SPATIAL INFORMATION !!!!!!!!!
print('Splitting data into TRAIN, VAL and TEST')
train_size = int(0.02 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) #-- pytorch alternative to the train_test_split command line from Scikit-Learn

train_size = int(0.5 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader
print("Training set size: ", len(train_dataset))
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size = config.BATCH_SIZE)
print("Validation set size: ", len(val_dataset))
val_dataset_loader = DataLoader(dataset=val_dataset, batch_size = config.BATCH_SIZE)
print("Testing set size: ", len(test_dataset))
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size = config.BATCH_SIZE)

# Initialize our model
model = model.unet_model.to(config.DEVICE)
# print(model)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=config.LR)

# calculate steps per epoch for training and test set
trainSteps = len(train_dataset) // config.BATCH_SIZE
testSteps = len(test_dataset) // config.BATCH_SIZE

# initialize a dictionary to store training history (keep track on training and testing lossses)
training_history = {"avg_train_loss": [], "train_accuracy": [], "IoU": [], "F1 score":[]}

# Training the network 
print('Training the network...🤗')
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    model.train()

    # For each epoch initialize the: 
    # total training and validation loss
    
    totalTrainLoss = 0
    total_val_Loss = 0
    # number of correctly classified pixels and the total number of pixels
    train_correct = 0
    total_n_pixels = 0
    target_true = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(train_dataset_loader):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        opt.zero_grad()  # zero out any previously accumulated gradients
        loss.backward() # obtain the gradients with respect to the loss
        opt.step() # perform one step of gradient descendent
        totalTrainLoss += loss  # add the loss to the total training loss so far

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        print(f'avg train loss {avgTrainLoss}')

        # Overall accuracy
        train_accuracy = metrics.pixel_acc(pred, y, train_correct, total_n_pixels)

        # IOU
        iou = metrics.jaccard_idx(pred, y)

        # F1 Score
        fscore = metrics.metrics(pred, y)[0]


    ## update training history
    training_history["avg_train_loss"].append(avgTrainLoss.cpu().detach().numpy()) # save the avg loss
    training_history["train_accuracy"].append(train_accuracy.cpu().detach().numpy()) # divided number of correct labels by total number of pixels in the batch
    training_history["IoU"].append(iou.cpu().detach().numpy()) 
    training_history["F1 score"].append(fscore.numpy()) 

    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))

print(training_history)

''' Implement other metrics https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388 '''


# Next Goals
# readjust tqdm
# Finish Training and validation loops
# Plot the accuracy and training loss
# Implement the remaining metrics
# Implemnt predictions     


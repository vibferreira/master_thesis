from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

import model
import metrics
import config
import split

import matplotlib.pyplot as plt

# Initialize our model
model = model.unet_model.to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=config.LR)

# calculate steps per epoch for training and test set
trainSteps = len(split.train_dataset) // config.BATCH_SIZE
valSteps = len(split.val_dataset) // config.BATCH_SIZE

# initialize a dictionary to store TRAINING history (keep track on training)
training_history = {"avg_train_loss": [], "train_accuracy": []}

# initialize a dictionary to store VALIDATION history (keep track on VALIDATION)
validation_history = {"avg_val_loss": [], "val_accuracy": []}

# Training the network 
print('Training the network...🤗')
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    model.train()

    # For each epoch initialize the: 
    # total training
    totalTrainLoss = 0
    totalValLoss = 0

    # number of correctly classified pixels and the total number of pixels
    train_correct = 0
    total_n_pixels = 0

    # loop over the training set
    loop = tqdm(split.train_dataloader, leave=False)
    for x, y in loop:
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        
        opt.zero_grad()  # zero out any previously accumulated gradients
        loss.backward() # obtain the gradients with respect to the loss
        opt.step() # perform one step of gradient descendent
        totalTrainLoss += loss  # add the loss to the total training loss so far
        
        # Accuracy
        # metrics.pixel_acc(pred, y:torch, 
        #       correct_pixels:int, 
        #       total_pixels:int)
        
        
    # switch off autograd
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x_val, y_val) in split.val_dataloader:
            # send the input to the device
            (x_val, y_val) = (x_val.to(config.DEVICE), y_val.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred_val = model(x_val)
            loss = lossFunc(pred_val, y_val)
            totalValLoss += loss
            print(loss)

    # calculate the average training and validation loss PER EPOCH
    avgTrainLoss = totalTrainLoss / trainSteps 
    avgValLoss = totalValLoss / valSteps
    
    print(f'avg train loss {avgTrainLoss}')
    print('val loss', avgValLoss)

    ## update training history
    training_history["avg_train_loss"].append(avgTrainLoss.cpu().detach().numpy()) # save the avg loss
    validation_history["avg_val_loss"].append(avgValLoss.cpu().detach().numpy()) # save the avg loss

    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))

print(training_history)
print(validation_history)
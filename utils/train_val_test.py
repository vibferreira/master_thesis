'''Training, Validation and Prediction functions'''

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid, save_image
from torch.cuda.amp import GradScaler, autocast

import segmentation_models_pytorch
import segmentation_models_pytorch as smp

import model
import metrics
import config
import utis

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb


import matplotlib.pyplot as plt

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def training(network, 
          trainloader, 
          optimizer, 
          loss_function, 
          save_path, 
          epoch,
          training_history, 
          scaler):
    
    current_loss = 0.0 # Set current loss value
    iou_train, acc = 0.0, 0.0  # metrics
    loop = tqdm(trainloader, leave=False) # training tracker
    
    # Iterate over the DataLoader for training data
    for i, (x, y) in enumerate(loop, 0):

        # Get inputs
        inputs, targets = (x.to(config.DEVICE), y.to(config.DEVICE))

        # training loop
        # optimizer.zero_grad() # Zero the gradients
        # outputs = network(inputs) # Perform forward pass
        # loss = loss_function(outputs, targets) # Compute loss
        # loss.backward()# Perform backward pass
        # optimizer.step() # Perform optimization
        
        # forward with autocast        
        with autocast():
            outputs = network(inputs)
            loss = loss_function(outputs, targets)
            
        optimizer.zero_grad()  # zero out any previously accumulated gradients    
        scaler.scale(loss).backward() # study this 
        scaler.step(optimizer)
        scaler.update()
        
        # total loss
        current_loss += loss.item()
        
        # metrics      
        all_metrics = metrics.metrics(outputs, targets)
        iou_train += all_metrics['iou']
        acc += all_metrics['acc']
        
        # update tqdm
        loop.set_description(f'Training Epoch [{epoch}/{config.NUM_EPOCHS}]')
        loop.set_postfix(loss_train=loss.item(), iou_train = iou_train, acc=acc)
    
    # averages per epoch
    avg_loss = current_loss / len(trainloader)
    avg_iou = (iou_train / len(trainloader))*100
    avg_acc = (acc / len(trainloader))*100
    
    # save
    training_history["avg_loss"].append(avg_loss) # save the avg loss
    training_history["accuracy"].append(avg_acc) # save the acc
    training_history["iou"].append(avg_iou) # save the acc
    
    # WANDB
    # wandb.log({
    # # "Examples": example_images,
    # "Train Loss": avg_loss,
    # "Train Accuracy": avg_acc,
    # "IoU_train":avg_iou})
    
    return training_history

def val(network, 
        testloader, 
        epoch,
        loss_function,
        validation_history,
        fold):
    
    current_loss = 0.0
    iou_val, acc = 0.0,0.0
    
    with torch.no_grad():
        loop = tqdm(testloader, leave=False)
        
        # Iterate over the test data and generate predictions
        for i, (x, y) in enumerate(loop, 0):

            # Get inputs
            inputs, targets = (x.to(config.DEVICE), y.to(config.DEVICE))

            # Generate outputs
            outputs = network(inputs)
            
            #loss
            loss = loss_function(outputs, targets) 
            current_loss += loss

            # metrics      
            all_metrics = metrics.metrics(outputs, targets)
            iou_val += all_metrics['iou']
            acc += all_metrics['acc']
            
            # update tqdm
            loop.set_description(f'Validation Epoch [{epoch}/{config.NUM_EPOCHS}]')
            loop.set_postfix(iou_val = iou_val, acc = acc)
        
    avgIOU = iou_val / len(testloader)
    avgLoss = current_loss / len(testloader)
    avgACC = acc / len(testloader)

    # Print accuracy
    # print('IoU for fold %d: %d %%' % (fold, 100.0 * avgIOU))
    # print('--------------------------------')
    results = 100.0 * (avgIOU)
    
    # store results
    validation_history["avg_val_loss"].append(avgLoss.cpu().detach().numpy()) # save the avg loss
    validation_history["val_accuracy"].append(avgACC) # save the acc
    validation_history["IoU_val"].append(avgIOU) # save the acc
    
    return results, validation_history
    
def make_predictions(model, 
                     dataloader, 
                     X_test,
                     folder,
                     print_pred=False, 
                     save_patches=False) -> list:
    model.eval()

    # save the predicons and the targets
    y_hat_test = []
    y_true_test = []
    y_score_test = []

    # retrieve X_test's ids
    file_idxs = [utis.get_file_index(ids) for ids in X_test]

    # get the image coords
    coords = utis.get_coords(X_test)

    # switch off autograd
    with torch.no_grad():
        # loop over the validation set
        for batch_idx, ((x_test, y_test), file_id) in enumerate(zip(dataloader, file_idxs)):
            (x_test[file_id], y_test[file_id]) = (x_test[file_id].to(config.DEVICE), y_test[file_id].to(config.DEVICE))

            # predictions
            pred_test = model(x_test[file_id])

            # Assign appropriate class 
            pred_test_class = (pred_test > 0.5).detach().float() # last layer is already sigmoid

            # Storing predictions and true labels 
            y_hat_test.append(pred_test_class.cpu().view(-1))
            y_true_test.append(y_test[file_id].cpu().view(-1).float())
            y_score_test.append(pred_test)

            # Plotting test       
            if print_pred:
                utis.plot_comparison(x_test[file_id], pred_test_class, y_test[file_id])
            plt.show()

            # Save patches 
            if save_patches:
                print(f'Saving patch_{batch_idx}_id_{file_id}.tif')
                utis.custom_save_patches(y_test[file_id], coords, file_id, batch_idx, folder, subfolder='masks') 
                utis.custom_save_patches(pred_test_class, coords, file_id, batch_idx, folder, subfolder='predictions') 

        # Stack and flatten for confusion matrix 
        y_hat_stack = torch.stack(y_hat_test)
        y_true_stack = torch.stack(y_true_test)
        y_score_stack = torch.stack(y_score_test)
        
        return y_hat_stack.view(-1), y_true_stack.view(-1), y_score_stack.view(-1)
            
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
def k_fold_dataloaders(train_idx, 
                    test_idx, 
                    dataset):
    # 
    torch.manual_seed(42)
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) 
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    # print('number of train samples', len(train_subsampler))

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=config.BATCH_SIZE, sampler=train_subsampler)

    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=config.BATCH_SIZE, sampler=test_subsampler)

    
    return trainloader, testloader
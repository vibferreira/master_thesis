'''Training, Validation and Prediction functions'''

from tqdm import tqdm
import numpy as np
import glob
import os

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

# def training(network, 
#           trainloader, 
#           optimizer, 
#           loss_function, 
#           save_path, 
#           epoch,
#           training_history, 
#           scaler):
    
#     # set the model in training mode
#     network.train()
    
#     current_loss = 0.0 # Set current loss value
#     iou_train, acc = 0.0, 0.0  # metrics
#     loop = tqdm(trainloader, leave=False) # training tracker
    
#     # Iterate over the DataLoader for training data
#     for i, (x, y) in enumerate(loop, 0):

#         # Get inputs
#         inputs, targets = (x.to(config.DEVICE), y.to(config.DEVICE))

#         # training loop
#         # optimizer.zero_grad() # Zero the gradients
#         # outputs = network(inputs) # Perform forward pass
#         # loss = loss_function(outputs, targets) # Compute loss
#         # loss.backward()# Perform backward pass
#         # optimizer.step() # Perform optimization
        
#         # forward with autocast        
#         with autocast():
#             outputs = network(inputs)
#             loss = loss_function(outputs, targets)
            
#         optimizer.zero_grad()  # zero out any previously accumulated gradients    
#         scaler.scale(loss).backward() # study this 
#         scaler.step(optimizer)
#         scaler.update()
        
#         # total loss
#         current_loss += loss.item()
        
#         # metrics      
#         all_metrics = metrics.metrics(outputs, targets)
#         iou_train += all_metrics['iou']
#         acc += all_metrics['acc']
        
#         print('iou',iou_train)
        
#         # update tqdm
#         loop.set_description(f'Training Epoch [{epoch}/{config.NUM_EPOCHS}]')
#         loop.set_postfix(loss_train=loss.item(), iou_train = iou_train, acc=acc)
    
#     # averages per epoch
#     avg_loss = current_loss / len(trainloader)
#     avg_iou = (iou_train / len(trainloader))*100
#     avg_acc = (acc / len(trainloader))*100
    
#     # save
#     training_history["avg_loss"].append(avg_loss) # save the avg loss
#     training_history["accuracy"].append(avg_acc) # save the acc
#     training_history["iou"].append(avg_iou) # save the acc
    
#     # WANDB
#     # wandb.log({
#     # # "Examples": example_images,
#     # "Train Loss": avg_loss,
#     # "Train Accuracy": avg_acc,
#     # "IoU_train":avg_iou})
    
#     return training_history

# def val(network, 
#         testloader, 
#         epoch,
#         loss_function,
#         validation_history,
#         fold):
    
#     # set the model in evaluation mode
#     network.eval()
    
#     current_loss = 0.0
#     iou_val, acc = 0.0,0.0

#     with torch.no_grad():
#         loop = tqdm(testloader, leave=False)
        
#         # Iterate over the test data and generate predictions
#         for i, (x, y) in enumerate(loop, 0):

#             # Get inputs
#             inputs, targets = (x.to(config.DEVICE), y.to(config.DEVICE))

#             # Generate outputs
#             outputs = network(inputs)
            
#             #loss
#             loss = loss_function(outputs, targets) 
#             current_loss += loss

#             # metrics      
#             all_metrics = metrics.metrics(outputs, targets)
#             iou_val += all_metrics['iou']
#             acc += all_metrics['acc']
            
#             print(iou_val)
#             # update tqdm
#             loop.set_description(f'Validation Epoch [{epoch}/{config.NUM_EPOCHS}]')
#             loop.set_postfix(iou_val = iou_val, acc = acc)
        
#     avgIOU = iou_val / len(testloader) * 100
#     avgLoss = current_loss / len(testloader)
#     avgACC = acc / len(testloader) * 100

#     # Print accuracy
#     # print('IoU for fold %d: %d %%' % (fold, 100.0 * avgIOU))
#     # print('--------------------------------')
#     results = 100.0 * (avgIOU)
    
#     # store results
#     validation_history["avg_val_loss"].append(avgLoss.cpu().detach().numpy()) # save the avg loss
#     validation_history["val_accuracy"].append(avgACC) # save the acc
#     validation_history["IoU_val"].append(avgIOU) # save the acc
    
#     return results, validation_history


def train(model, dataloader, opt, lossFunc, epoch, scaler, training_history):
    # set the model in training mode
    model.train()

    # Save total train loss
    totalTrainLoss = 0
    
    # metrics
    accuracy = 0
    iou = 0
    f1score = 0
    dice = 0
    
    # loop over the training set
    loop = tqdm(dataloader, leave=False)
    for x, y in loop:
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.float().to(config.DEVICE))
        
#         # forward with autocast        
#         with autocast():
#             pred = model(x)
#             loss = lossFunc(pred, y)
            
#         optim.zero_grad()  # zero out any previously accumulated gradients    
#         scaler.scale(loss).backward() # study this 
#         scaler.step(optim)
#         scaler.update()
        
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        
        opt.zero_grad()  # zero out any previously accumulated gradients
        loss.backward() # obtain the gradients with respect to the loss
        opt.step() # perform one step of gradient descendent
        
        totalTrainLoss += loss  # add the loss to the total training loss so far 
        
        # metrics      
        all_metrics = metrics.metrics(pred, y)
        accuracy += all_metrics['acc']
        iou += all_metrics['iou']
        f1score += all_metrics['f1score']
        dice += all_metrics['dice_coeff']
        
        # update tqdm
        loop.set_description(f'Training Epoch [{epoch}/{config.NUM_EPOCHS}]')
        loop.set_postfix(loss=loss.item(), acc = all_metrics['acc'], iou=all_metrics['iou'], dice = all_metrics['dice_coeff'])
        
    # calculate the average training loss PER EPOCH
    avgTrainLoss = totalTrainLoss / len(dataloader)
    avgAccLoss = accuracy / len(dataloader)
    avgIOU = iou / len(dataloader)
    avgF1score = f1score / len(dataloader)
    avgDice = dice / len(dataloader)
    
    ## update training history
    training_history["avg_train_loss"].append(avgTrainLoss.cpu().detach().numpy()) # save the avg loss
    training_history["train_accuracy"].append(avgAccLoss) # save the acc 
    training_history["IoU"].append(avgIOU) # save the acc 
    
    # WANDB
    wandb.log({
    # "Examples": example_images,
    "Train Loss": avgTrainLoss,
    "Train Accuracy": avgAccLoss,
    "Train IOU":avgIOU})
    
    return training_history

def validation(model, dataloader, lossFunc, epoch, validation_history):
    
    # set the model in evaluation mode
    model.eval()
    # Save total train loss
    totalValLoss = 0
    
    # metrics
    accuracy_val = 0
    iou_val = 0
    f1score_val = 0
    
    example_pred = []
    example_gt = []
    # switch off autograd
    with torch.no_grad():
        # loop over the validation set
        loop = tqdm(dataloader, leave=False)
        
        for batch_idx, (x_val, y_val) in enumerate(loop):
            # send the input to the device
            (x_val, y_val) = (x_val.to(config.DEVICE), y_val.to(config.DEVICE))
            
            # make the predictions and calculate the validation loss
            pred_val = model(x_val)
            loss = lossFunc(pred_val, y_val)
            totalValLoss += loss
            
            # metrics      
            all_metrics = metrics.metrics(pred_val, y_val)
            accuracy_val += all_metrics['acc']
            iou_val += all_metrics['iou']
            f1score_val += all_metrics['f1score']

            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            # example_pred.append(wandb.Image(pred_val[0], caption=f"pred_iter_n_{batch_idx}"))
            # example_gt.append(wandb.Image(y_val[0].float(), caption=f"gt_iter_n_{batch_idx}"))
            
            # update tqdm
            loop.set_description(f'Validation Epoch [{epoch}/{config.NUM_EPOCHS}]')
            loop.set_postfix(loss_val=loss.item(), acc_val = all_metrics['acc'], iou_val=all_metrics['iou'])
                        
    # calculate the average VALIDATION loss PER EPOCH
    avgValLoss = totalValLoss / len(dataloader)
    avgAccLoss = accuracy_val / len(dataloader)
    avgIOU = iou_val / len(dataloader)
    avgF1score = f1score_val / len(dataloader)

    ## update VALIDATION history
    validation_history["avg_val_loss"].append(avgValLoss.cpu().detach().numpy()) # save the avg loss
    validation_history["val_accuracy"].append(avgAccLoss) # save the acc
    validation_history["IoU_val"].append(avgIOU) # save the acc
    
    # WANDB
    wandb.log({
    # "Predictions": example_pred,
    # "GT": example_gt,
    "Val Accuracy": avgAccLoss,
    "Val Loss": avgValLoss,
    "Val IOU": avgIOU})
    
    return validation_history
    
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
    
    # delete files in the folder
    filelist = glob.glob(f"{folder}/{'masks'}" +'/*.tif')
    filelist_2 = glob.glob(f"{folder}/{'predictions'}" +'/*.tif')
    [os.remove(f) for f in filelist]
    [os.remove(f) for f in filelist_2]
    
    print(filelist)
    
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
    train_subsampler = torch.utils.data.Subset(dataset, train_idx) 
    test_subsampler = torch.utils.data.Subset(dataset, test_idx)
    # print('number of train samples', len(train_subsampler))

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      train_subsampler, 
                      batch_size=config.BATCH_SIZE,
                      shuffle = True
                      # sampler=train_subsampler
    )

    testloader = torch.utils.data.DataLoader(
                      test_subsampler,
                      batch_size=config.BATCH_SIZE,
                      shuffle = True
                      # sampler=test_subsampler
    )

    return trainloader, testloader
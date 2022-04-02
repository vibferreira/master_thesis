'''Defines utils functions to save, plot and make predictions  '''
''' Functions to plot using torchvision make_grid are from https://medium.com/analytics-vidhya/do-you-visualize-dataloaders-for-deep-neural-networks-7840ae58fee7'''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import segmentation_models_pytorch
from tqdm import tqdm

import model
import metrics
import config
import utis
import train

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb


import torch

def plot_comparison(x:torch.Tensor, 
                    pred:torch.Tensor, 
                    y:torch.Tensor) -> None:
    
    gt = np.squeeze(y.data.cpu().cpu().numpy()[0])
    pred = np.squeeze(pred.sigmoid().cpu().numpy())
    img = np.squeeze(x.data.cpu().cpu().numpy()[0])
    _, ax = plt.subplots(1, 3, sharey='row')
    
    plt.gray()
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(gt)
    ax[1].set_title('Ground Truth')
    ax[2].imshow(pred[0])
    ax[2].set_title('Prediction')
    plt.show()

def save_best_model(model, 
                    dest_path: str, 
                    val_dic:dict, 
                    e) -> None:
    
    ''' Saves the best model
    Args: 
    model (class): instance of the the model class
    dest_path (str): destination path
    val_dict (dict): dictionary storing valdation accuracies'''
    
    iou = float(val_dic['IoU_val'][-1])
    acc = float(val_dic['val_accuracy'][-1])
    [os.remove(f) for f in glob.glob(dest_path + '/*')] # remove previous saved files 
    return torch.save(model.state_dict(), dest_path + f'/best_model_epoch_{e +1}_iou_{round(iou,3)}_acc_{round(acc,3)}.pth')


def plot_grids(grids, titles = ["Input", 'Target']):
    nrow = len(grids)
    fig = plt.figure(figsize=(8, nrow), dpi=300)
    # Remove the space between the grids
    fig.subplots_adjust(wspace=0, hspace=0)
    # Each grid is new subplot
    for i in range(1,nrow+1):
        sub = fig.add_subplot(nrow,1,i)
        # Dont show the x-axis
        sub.xaxis.set_visible(False)
        # Remove y axis ticks and set yaxis label
        sub.set_yticks([])
        sub.set_ylabel(titles[i-1], rotation=0, fontsize=5, labelpad=15)
        sub.imshow(grids[i-1])
    plt.show()

def create_grids(imgList, nrofItems, pad, norm = True):
    '''Create a list of Image grids from Torch tensors'''
    # Permute the axis as numpy expects image of shape (H x W x C) 
    return list(map(lambda imgs: make_grid(imgs[:nrofItems], normalize=norm, padding = pad, nrow=nrofItems).permute(1, 2, 0), imgList))

# Create list of two image grids - Input Images & Target Mask Images for Segmentation task
def create_segement_grids(loader_iter, nrofItems = 5, pad = 4):
    '''# Create list of two image grids - Input Images & Target Mask Images for Segmentation task'''
    inp, target = next(loader_iter)
    return create_grids([inp, target], nrofItems, pad)

def make_predictions(model:segmentation_models_pytorch.unet.model.Unet, 
                     dataloader:torch.utils.data.dataloader.DataLoader) -> list:
    model.eval()

    # Save total train loss
    totalValLoss = 0

    # log the predictions to WANDB
    example_pred = []
    example_gt = []

    # save the predicons and the targets
    y_hat_test = []
    y_true_test = []

    # switch off autograd
    with torch.no_grad():
        # loop over the validation set
        loop = tqdm(dataloader, leave=False)

        for batch_idx, (x_test, y_test) in enumerate(loop):
            # send the input to the device
            (x_test, y_test) = (x_test.to(config.DEVICE), y_test.to(config.DEVICE))

            # predictions
            pred_test = model(x_test)

            # Assign appropriate class 
            pred_test = (pred_test > 0.5).float() # last layer is already sigmoid

            # Storing predictions and true labels 
            y_hat_test.append(pred_test.cpu().view(-1, ))
            y_true_test.append(y_test.cpu().view(-1, ).float())

            # # Plotting test
            # utis.plot_comparison(x_test, pred_test, y_test)

            # # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            example_pred.append(wandb.Image(pred_test[0], caption=f"pred_iter_n_{batch_idx}"))
            example_gt.append(wandb.Image(y_test[0].float(), caption=f"gt_iter_n_{batch_idx}"))

            # update tqdm
            loop.set_description(f'Testing Epoch')
            
            # Save images
            # print(f'Saving {pred_{idx}.png}')
            # save_image(pred_test, f"{folder}/pred_{idx}.png") 
            # save_image(y_test, f"{folder}/y_true_{idx}.png")

        # WANDB
        wandb.log({
        "Predictions": example_pred,
        "GT": example_gt})

        # Stack and flatten for confusion matrix # GETTING SIZE ERROR AT THE MOMENT
        y_hat_stack = torch.stack(y_hat_test)
        y_true_stack = torch.stack(y_true_test)
        
        return y_hat_stack, y_true_stack
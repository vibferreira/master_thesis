''' This file contains all the metrics used to evaluate the model'''
# make it as a class later

import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np


def pixel_accuracy(pred:torch, 
                   y:torch) -> int:
    # implementation from https://www.kaggle.com/ligtfeather/semantic-segmentation-is-easy-with-pytorch#Training
        
    ''' Calculates pixel Accuracy
    Args:
    pred (torch): output from the model
    y (torch) : mask 
    Results:
    int: accuracy
    '''  
    with torch.no_grad(): # no need of gradients 
        # pred = torch.argmax(F.softmax(pred, dim=0), dim=0) # get the prediction 
        pred = pred.sigmoid()
        pred = (pred > 0.5).float()

        correct = torch.eq(pred, y).int() # checks if output == mask
        accuracy = float(correct.sum()) / float(correct.numel()) # divide by the total number of pixels in the input tensor
    return accuracy

# IOU / Jaccard
def jaccard_idx(pred:torch, 
                y:torch) -> np.array:
    ''' Calculates Jaccard Index
    Args:
    pred (torch): output from the model
    y (torch) : mask 
    Results:
    torch: 
    '''
    # Calculate IoU
    # pred = torch.argmax(F.softmax(pred, dim=0), dim=0) # get the preds
    pred = pred.sigmoid()
    pred = (pred > 0.5).float()
    
    intersection = torch.logical_and(pred, y)
    union = torch.logical_or(pred,y)
    return (torch.sum(intersection) / torch.sum(union)).numpy() # return the intersection over union

# F1 Score
def metrics(pred, y):

    pred = pred.detach()
    pred = pred.sigmoid()
    pred = (pred > 0.5).float()
    # # pred = pred.view(-1, )
    # y = y.view(-1, ).float()

    tp = torch.sum(pred * y)  # TP
    fp = torch.sum(pred * (1 - y))  # FP
    fn = torch.sum((1 - pred) * y)  # FN
    tn = torch.sum((1 - pred) * (1 - y))  # TN

    eps=1e-5
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    iou = (tp + eps) / (fp + tp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    f1score = 2 * precision * recall / (precision + recall)

    return {'acc': pixel_acc, 'iou':iou, 'dice_coeff': dice, 'f1score': f1score}

# Confusion metrics 

# # assert the dimensions first
# assert pred.shape == y.shape, f'Shape of pred is {pred.shape} and shape of y is {y.shape}'
''' This file contains all the metrics used to evaluate the model'''
# make it as a class later

import torch
from sklearn.metrics import f1_score

def pixel_acc(pred:torch, 
              y:torch, 
              correct_pixels:int, 
              total_pixels:int) -> float:

    ''' Calculates overall accuracy
    Args:
    pred (torch): model predictions,
    y (torch): ground truth,
    train_correct (int): number of correctly classified pixels
    total_n_pixels (int): total number of pixels
    Returns:
    float: percentage of correctly classified pixels'''

    # assert the dimensions first
    assert (pred.shape == y.shape, f'Shape of pred is {pred.shape} and shape of y is {y.shape}')

    # Calculate accuracy
    y_hat_class = torch.argmax(pred.detach(), axis=0) # assign a label based on the network's prediction. Axis zero is taking image by image
    correct_pixels += torch.sum(y_hat_class==y).float() # number of correctly classified pixels (defined in the loop)
    total_pixels += y.nelement() # seems to be correct (defined in the loop)

    return 100*correct_pixels/total_pixels

# IOU
def jaccard_idx(pred:torch, 
                y:torch) -> torch:
    ''' Calculates Jaccard Index
    Args:
    pred (torch): output from the model
    y (torch) : mask 
    Results:
    torch: 
    '''
    # assert the dimensions first
    assert (pred.shape == y.shape, f'Shape of pred is {pred.shape} and shape of y is {y.shape}')

    # Calculate IoU
    intersection = torch.logical_and(pred, y)
    union = torch.logical_or(pred,y)
    return torch.sum(intersection) / torch.sum(union) # return the intersection over union

# F1 Score
def metrics(pred, y):

    pred = pred.detach()
    # # pred = pred.view(-1, )
    # y = y.view(-1, ).float()

    tp = torch.sum(pred * y)  # TP
    fp = torch.sum(pred * (1 - y))  # FP
    fn = torch.sum((1 - pred) * y)  # FN
    tn = torch.sum((1 - pred) * (1 - y))  # TN

    eps=1e-5
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    f1score = 2 * precision * recall / (precision + recall)

    return f1score, dice,  pixel_acc

# Confusion metrics 
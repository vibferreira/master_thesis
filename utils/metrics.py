''' This file contains all the metrics used to evaluate the model'''
# make it as a class later

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix, classification_report

import seaborn as sns

from matplotlib import pyplot as plt, cm
from matplotlib.ticker import PercentFormatter

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

# Jaccard 
def IoU(inputs, targets, smooth=1):
        
        # #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        bs = targets.size(0)
        targets = targets.view(bs, 1, -1)
        inputs = inputs.view(bs, 1, -1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
# Dice
def Dice(inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# F1 Score
def metrics(pred: torch.Tensor, 
            y: torch.Tensor):
    
    pred = pred.detach() # detach from the grads
    pred = (pred > 0.5).float() # classify into 0 and 1 
    
    # pred = pred.view(-1, )
    # y = y.view(-1, ).float()
    
    bs = y.size(0)
    y = y.view(bs, 1, -1).float()
    pred = pred.view(bs, 1, -1)
    
    tp = torch.sum(torch.abs(pred * y))  # TP
    fp = torch.sum(torch.abs(pred * (1 - y)))  # FP
    fn = torch.sum(torch.abs((1 - pred) * y))  # FN
    tn = torch.sum(torch.abs((1 - pred) * (1 - y)))  # TN
    
    eps = 1e-5 # avoid division by 0
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    f1score = 2 * precision * recall / (precision + recall)

    return {'false_preds': np.array([tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), tn.cpu().numpy()]),
            'acc': pixel_acc.cpu().numpy(), 
            'iou':iou.cpu().numpy(), 
            'dice_coeff': dice.cpu().numpy(), 
            'f1score': f1score.cpu().numpy()}
    

def cm_analysis(y_true, y_pred, labels, classes, figsize=(6,4)):
    """
    !!!!!!!!!!!!!!!
    Adapted from Mesquita : https://gist.github.com/mesquita/f6beffcc2579c6f3a97c9d93e278a9f1
    !!!!!!!!!!!!!!!!!
    
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=1)
#     plt.rc('font', size=20)
#     plt.rc('axes', titlesize=20)
#     font sizes
    size=15
    params = {'font.size':size,
              'legend.fontsize': 'large',
              'figure.figsize': (25,10),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.75,
              'ytick.labelsize': size*0.75,
              'axes.titlepad': 25}
    plt.rcParams.update(params)
    
    # reshape tensor
    y_true = y_true.squeeze(0).view(-1)
    y_pred = y_pred.squeeze(0).view(-1)

    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
                
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    # sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
    sns_plot=sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
    # plt.savefig(filename,  bbox_inches='tight')
    ax.set_title('Confusion Matrix')
    plt.show()
    
    # save
    sns_plot.figure.savefig("output.png")
    
    # classification report
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # testing approach
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    
    # IOU or DICE 
    all_metrics = metrics(y_pred, y_true)
    iou = all_metrics['iou']
    f1score = all_metrics['f1score']
    
    print('IoU', iou)
    print('F1 score', f1score)
    



    

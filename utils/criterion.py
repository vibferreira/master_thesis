import torch.nn as nn
from utils import config

''' STUDY PURPOSES, IMPLEMENTING LOSS FUNCTIONS'''

def cross_entropy_loss(y_hat, y):
    return nn.CrossEntropyLoss()(y_hat, y) # in relation to the class probability 


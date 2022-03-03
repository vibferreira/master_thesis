import torch.nn as nn

def loss_func(y_hat, y):
    return nn.CrossEntropyLoss()(y_hat, y) # in relation to the class probability 


''' Implementation from https://github.com/clemkoa/u-net'''

import torch
import config
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

unet_model = smp.Unet(
    encoder_name= config.BACKBONE,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=config.N_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes= config.N_CLASSES,    # model output channels (number of classes in your dataset)
)
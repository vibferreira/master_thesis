''' Implementation from https://github.com/clemkoa/u-net'''
import config
import segmentation_models_pytorch as smp

unet_model = smp.Unet(
    encoder_name= config.BACKBONE,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=config.N_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes= config.N_CLASSES, # model output channels (number of classes in your dataset)
    activation = 'sigmoid', # activation function, default is None
    decoder_attention_type = config.ATT # None or'scse' Attention module used in decoder of the model. Available options are **None** and **scse** (https://arxiv.org/abs/1808.08127)
)
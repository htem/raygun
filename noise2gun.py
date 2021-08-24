# !conda activate n2v
import numpy as np
from matplotlib import pyplot as plt
import torch
import zarr
import os

from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import logging
logging.basicConfig(level=logging.INFO)
# from tqdm.auto import trange

# from this repo
import loser
from boilerPlate import BoilerPlate
# from segway.tasks.make_zarr_from_tiff import task_make_zarr_from_tiff_volume as tif2zarr

class Noise2Gun():

    def __init__(self,
            train_source, #EXPECTS ZARR VOLUME
            voxel_size,
            model_name='noise2gun',
            model_path='./models/',
            side_length=64,#12 # in voxels for prediction (i.e. network output) - actual used ROI for network input will be bigger for valid padding
            unet_depth=4, # number of layers in unet
            downsample_factor=2,
            conv_padding='valid',
            num_fmaps=12,
            fmap_inc_factor=5,
            perc_hotPixels=0.198,
            constant_upsample=True,
            num_epochs=10000,
            batch_size=1,
            init_learning_rate=1e-5,#0.0004#1e-6 # init_learn_rate = 0.0004
            log_every=100,
            tensorboard_path='./tensorboard/'
            ):
            self.train_source = train_source
            self.voxel_size = voxel_size
            self.model_name = model_name
            self.model_path = model_path
            self.side_length = side_length
            self.unet_depth = unet_depth
            self.downsample_factor = downsample_factor
            self.conv_padding = conv_padding
            self.num_fmaps = num_fmaps
            self.fmap_inc_factor = fmap_inc_factor
            self.perc_hotPixels = perc_hotPixels
            self.constant_upsample = constant_upsample
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            self.init_learning_rate = init_learning_rate
            self.log_every = log_every
            self.tensorboard_path = tensorboard_path
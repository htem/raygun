import numpy as np
import torch
import random
import logging

import gunpowder as gp
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch, tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain

from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)

from skimage import filters

class BoilerPlate(gp.BatchFilter):
    def __init__(self, raw_array, mask_array, hot_array, plate_size=None, perc_hotPixels = 0.198):
        self.raw_array = raw_array
        self.mask_array = mask_array
        self.hot_array = hot_array
        self.plate_size = plate_size
        self.perc_hotPixels = perc_hotPixels

    def setup(self):
        # tell downstream nodes about the new arrays
        self.provides(
            self.mask_array,
            self.spec[self.raw_array].copy())
        self.provides(
            self.hot_array,
            self.spec[self.raw_array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.raw_array] = request[self.hot_array].copy() # make sure we're getting the needed data (should already be requested, but just in case)
        return deps

    def process(self, batch, request):
        # make heat mask
        mask_data, hot_data = self.get_heat(batch)

        # create the array spec for the new arrays
        mask_spec = batch[self.raw_array].spec.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        mask_spec.roi = request[self.mask_array].roi.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        hot_spec = batch[self.raw_array].spec.copy() #TODO: DETERMINE IF THIS IS NECESSARY
        hot_spec.roi = request[self.hot_array].roi.copy() #TODO: DETERMINE IF THIS IS NECESSARY

        # create a new batch to hold the new array
        batch = gp.Batch()

        # create a new array
        mask = gp.Array(mask_data, mask_spec)
        hot = gp.Array(hot_data, hot_spec)

        # store it in the batch
        batch[self.mask_array] = mask
        batch[self.hot_array] = hot

        # return the new batch
        return batch

    def get_heat(self, batch):
        ndims = batch[self.raw_array].roi.dims()
        raw_data = batch[self.raw_array].data
        if not isinstance(self.plate_size, type(None)):
            if len(self.plate_size) > 1:
                plate_shape = self.plate_size
            else:
                plate_shape = [self.plate_size,]*ndims           
            pad_width = (np.array(raw_data.shape) - np.array(plate_shape)) // 2
            plate_volume = np.prod(plate_shape)
        else:
            pad_width = (0,)*ndims
            plate_volume = np.prod(raw_data.shape)
            plate_shape = raw_data.shape
        
        numPix = int(plate_volume *  self.perc_hotPixels)
        coords = self.getStratifiedCoords(numPix, plate_shape)

        mask = torch.zeros_like(raw_data) > 0
        hot = raw_data.copy()
        hot_pixels = torch.utils.data.RandomSampler(raw_data, replacement=True, num_samples=numPix)
        for i, coord in enumerate(coords):
            this_coord = tuple(np.add(coord, pad_width))
            mask[this_coord] = True
            hot[this_coord] = hot_pixels[i]

        return mask, hot

    def getStratifiedCoords(numPix, shape):
        '''
        Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
        '''
        ndims = len(shape)
        box_size = np.round((np.prod(shape) / numPix)**(1/ndims)).astype(np.int)
        coords = []
        box_counts = int(np.ceil(shape / box_size))
        
        rands = torch.tensor(np.random.randint(0, box_size, (np.prod(box_counts), ndims)))
        
        tensors = []
        for count in box_counts:
            tensors.append(torch.tensor(range(count)))        
        
        offset_tensors = torch.meshgrid(tensors)
        
        offsets = torch.zeros((len(offset_tensors[0].flatten()), ndims))
        for i, tensor in enumerate(offset_tensors):
            offsets[:, i] = tensor.flatten()
        
        temp_coords = offsets + rands
        coords = []
        for i in range(temp_coords.shape[0]):
            include = True
            for l, lim in enumerate(shape):
                include = include and (lim > temp_coords[i, l])
            if include:
                coords.append(tuple(temp_coords[i, :]))
        return coords
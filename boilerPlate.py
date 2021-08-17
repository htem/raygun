import numpy as np
import torch
import random
import logging
from skimage import filters
from skimage.util import *

import gunpowder as gp
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import torch, tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain

from typing import Dict, Union, Optional

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

from skimage import filters

class BoilerPlate(gp.BatchFilter):
    def __init__(self, raw_array, mask_array, hot_array, plate_size=None, perc_hotPixels = 0.198, ndims=3):
        self.raw_array = raw_array
        self.mask_array = mask_array
        self.hot_array = hot_array
        self.plate_size = plate_size #TODO: CAN BE INFERRED
        self.perc_hotPixels = perc_hotPixels
        self.ndims = ndims

    def setup(self):
        # tell downstream nodes about the new arrays
        mask_spec = self.spec[self.raw_array].copy()
        mask_spec.dtype = bool
        hot_spec = self.spec[self.raw_array].copy()
        self.provides(
            self.mask_array,
            mask_spec)
        self.provides(
            self.hot_array,
            hot_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.raw_array] = request[self.hot_array].copy() # make sure we're getting the needed data 
        return deps

    def process(self, batch, request):
        # make heat mask
        mask_data, hot_data = self.boil(batch)

        # create the array spec for the new arrays
        mask_spec = batch[self.raw_array].spec.copy() 
        mask_spec.dtype = bool
        hot_spec = batch[self.raw_array].spec.copy()
        hot_spec.dtype = hot_data.dtype

        # create a new batch to hold the new array
        new_batch = gp.Batch()

        # create a new array
        mask = gp.Array(mask_data, mask_spec)
        mask = mask.crop(request[self.mask_array].roi)
        hot = gp.Array(hot_data, hot_spec)

        # store it in the batch
        new_batch[self.mask_array] = mask
        new_batch[self.hot_array] = hot
        return new_batch

    def boil(self, batch):
        raw_data = batch[self.raw_array].data

        if not isinstance(self.plate_size, type(None)):
            if len(np.array(self.plate_size).flatten()) > 1:
                self.plate_shape = self.plate_size
            else:
                self.plate_shape = [self.plate_size,]*self.ndims           
            self.pad_width = (np.array(raw_data.shape[-self.ndims:]) - np.array(self.plate_shape)) // 2
        else:
            self.plate_shape = raw_data.shape[-self.ndims:]
            self.pad_width = (0,)*self.ndims
        
        self.numPix = int(np.prod(self.plate_shape) * self.perc_hotPixels)
        mask = np.zeros_like(raw_data) > 0
        hot = raw_data.copy()
        mask, hot = self.rec_heater(raw_data, mask, hot)
        return mask, hot
    
    def get_heat(self, raw_data, mask, hot):
        coords = self.getStratifiedCoords()        
        hot_pixels = np.random.choice(raw_data.flatten(), size=self.numPix)
        for i, coord in enumerate(coords):
            this_coord = tuple(np.add(coord, self.pad_width).astype(int))
            mask[this_coord] = True
            hot[this_coord] = hot_pixels[i]
        return mask, hot

    def rec_heater(self, raw_data, mask, hot):
        if len(raw_data.shape) > self.ndims:
            for i, data, this_mask, this_hot in enumerate(zip(raw_data, mask, hot)):
                mask[i], hot[i] = self.rec_heater(data, this_mask, this_hot)
            return mask, hot
        else:
            return self.get_heat(raw_data, mask, hot)

    def getStratifiedCoords(self):
        '''
        Produce a list of approx. 'numPix' random coordinate, sampled from 'shape' using startified sampling.
        '''
        box_size = np.round((1/self.perc_hotPixels)**(1/self.ndims)).astype(np.int)
        coords = []
        box_counts = np.ceil(self.plate_shape / box_size).astype(int)
        
        rands = torch.tensor(np.random.randint(0, box_size, (np.prod(box_counts), self.ndims)))
        
        tensors = []
        for count in box_counts:
            tensors.append(torch.tensor(range(count)))        
        
        offset_tensors = torch.meshgrid(tensors)
        
        offsets = torch.zeros((len(offset_tensors[0].flatten()), self.ndims))
        for i, tensor in enumerate(offset_tensors):
            offsets[:, i] = tensor.flatten()
        
        temp_coords = offsets + rands
        coords = []
        for i in range(temp_coords.shape[0]):
            include = True
            for l, lim in enumerate(self.plate_shape):
                include = include and (lim > temp_coords[i, l])
            if include:
                coords.append(tuple(temp_coords[i, :]))
        return coords

class GaussBlur(gp.BatchFilter):

    def __init__(self, array, sigma):
        self.array = array
        self.sigma = sigma
        self.truncate = 4
    
    def setup(self):
        # tell downstream nodes about the new arrays
        spec = self.spec[self.array].copy()
        spec.dtype = np.float64
        self.updates(
            self.array,
            spec)

    def prepare(self, request):

        # the requested ROI for array
        roi = request[self.array].roi

        # 1. compute the context
        context = gp.Coordinate((self.truncate,)*roi.dims()) * self.sigma

        # 2. enlarge the requested ROI by the context
        context_roi = roi.grow(context, context)
        context_roi = request[self.array].roi.intersect(context_roi) # make sure it doesn't go out of bounds

        # create a new request with our dependencies
        deps = gp.BatchRequest()
        deps[self.array] = context_roi

        # return the request
        return deps

    def process(self, batch, request):

        # 3. smooth the whole array (including the context)
        data = batch[self.array].data
        data = filters.gaussian(
        data,
        sigma=self.sigma,
        truncate=self.truncate)

        # 4. make sure to match original datatype
        # if data.dtype != batch[self.array].spec.dtype:
        #     if batch[self.array].spec.dtype == 'uint8':
        #         data = img_as_ubyte(data)
        #     elif batch[self.array].spec.dtype == 'uint16':
        #         data = img_as_uint(data)
        #     elif batch[self.array].spec.dtype == 'int16':
        #         data = img_as_int(data)
        #     elif batch[self.array].spec.dtype == 'float64':
        #         data = img_as_float(data)
        batch[self.array].spec.dtype = data.dtype
        batch[self.array].data = data

        # 5. crop the array back to the request
        batch[self.array] = batch[self.array].crop(request[self.array].roi)
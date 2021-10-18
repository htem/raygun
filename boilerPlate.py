import numpy as np
import torch
import random
import logging
from skimage import filters
from skimage.util import *

import gunpowder as gp

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
        #initialize stuff to be set on first call
        self.offsets = None
        self.plate_shape = None

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
        mask_data, hot_data = self.boil(batch, request)

        # create the array spec for the new arrays
        mask_spec = batch[self.raw_array].spec.copy()
        #mask_spec.roi = request[self.mask_array].roi.copy()
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

    def boil(self, batch, request):
        if self.plate_shape is None:
            hot_data_shape = batch[self.raw_array].data.shape
            if self.plate_size is not None:
                if len(np.array(self.plate_size).flatten()) > 1:
                    self.plate_shape = self.plate_size
                else:
                    self.plate_shape = [self.plate_size,]*self.ndims           
                self.pad_width = gp.Coordinate((np.array(hot_data_shape[-self.ndims:]) - np.array(self.plate_shape)) // 2)
            elif self.plate_shape:
                self.plate_shape = hot_data_shape[-self.ndims:]
                self.pad_width = gp.Coordinate((0,)*self.ndims)
            self.numPix = int(np.prod(self.plate_shape) * self.perc_hotPixels)
        raw_data_small = batch[self.raw_array].crop(request[self.mask_array].roi).data.copy()
        raw_data_big = batch[self.raw_array].data.copy()
        mask = np.zeros_like(raw_data_big) > 0
        #mask = np.zeros_like(raw_data_small) > 0
        hot = raw_data_big
        mask, hot = self.rec_heater(raw_data_small, mask, hot)
        return mask, hot
    
    def get_heat(self, raw_data, mask, hot):
        coords = self.getStratifiedCoords()        
        hot_pixels = np.random.choice(raw_data.flatten(), size=self.numPix)
        for i, coord in enumerate(coords):
            this_coord = coord + self.pad_width
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
        if self.offsets is None: # SET ON FIRST PASS (will break if tried again with different size inputs)
            #TODO: DETERMINE IF THIS IS SLOWING THINGS DOWN?
            self.box_size = np.round((1/self.perc_hotPixels)**(1/self.ndims)).astype(np.int)
            self.box_counts = np.ceil(self.plate_shape / self.box_size).astype(int)
        
            tensors = []
            for count in self.box_counts:
                tensors.append(torch.tensor(range(count)) * self.box_size)        
            
            offset_tensors = torch.meshgrid(tensors)
            
            self.offsets = torch.zeros((len(offset_tensors[0].flatten()), self.ndims))
            for i, tensor in enumerate(offset_tensors):
                self.offsets[:, i] = tensor.flatten()
        
        rands = torch.tensor(np.random.randint(0, self.box_size, (np.prod(self.box_counts), self.ndims)))
        
        temp_coords = self.offsets + rands
        coords = []
        for i in range(temp_coords.shape[0]): #TODO: SPEED THIS UP
            include = True
            for l, lim in enumerate(self.plate_shape):
                include = include and (lim > temp_coords[i, l])
            if include:
                coords.append(gp.Coordinate(temp_coords[i, :]))
        return coords

class GaussBlur(gp.BatchFilter):

    def __init__(self, array, sigma, truncate=4, new_array=None):
        self.array = array
        self.sigma = sigma
        self.new_array=new_array
        self.truncate = truncate
    
    def setup(self):
        self.enable_autoskip()
        if self.new_array is None:
            self.updates(self.array, self.spec[self.array])
        else:
            self.provides(self.new_array, self.spec[self.array].copy())

    def prepare(self, request):
        if self.new_array is None:
            this_array = self.array
        else:
            this_array = self.new_array
        # the requested ROI for array
        roi = request[this_array].roi

        # 1. compute the context
        context = gp.Coordinate((self.truncate,)*roi.dims()) * self.sigma

        # 2. enlarge the requested ROI by the context
        context_roi = roi.grow(context, context)
        context_roi = request[this_array].roi.intersect(context_roi) # make sure it doesn't go out of bounds
        
        # create a new request with our dependencies
        deps = gp.BatchRequest()
        deps[self.array] = context_roi

        # return the request
        return deps

    def process(self, batch, request):

        # smooth the whole array (including the context)
        data = batch[self.array].data.copy()
        data = filters.gaussian(
                            data,
                            sigma=self.sigma,
                            truncate=self.truncate)

        if self.new_array is None:
            batch[self.array].data = data
            # crop the array back to the request
            batch[self.array] = batch[self.array].crop(request[self.array].roi)
        else:
            new_batch = gp.Batch()
            blurred = gp.Array(data, batch[self.array].spec.copy())
            new_batch[self.new_array] = blurred.crop(request[self.new_array].roi)
            
            return new_batch

class Noiser(gp.BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        seed (``int``):

            Optionally set a random seed, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation

        new_array (:class:`ArrayKey`):

            New array (if any) to store result in, instead of array
    '''

    def __init__(self, array, mode='gaussian', seed=None, clip=True, new_array=None, **kwargs):
        self.array = array
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.new_array = new_array
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        if self.new_array is None:
            self.updates(self.array, self.spec[self.array])
        else:
            self.provides(self.new_array, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        if self.new_array is None:
            deps[self.array] = request[self.array].copy()
        else:
            deps[self.array] = request[self.new_array].copy()

        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Noise augmentation requires float types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
        assert raw.data.min() >= -1 and raw.data.max() <= 1, "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."
        if self.new_array is None:
            raw.data = random_noise(
                            raw.data,
                            mode=self.mode,
                            seed=self.seed,
                            clip=self.clip,
                            **self.kwargs).astype(raw.data.dtype)
        else:
            noised_data = random_noise(
                            raw.data,
                            mode=self.mode,
                            seed=self.seed,
                            clip=self.clip,
                            **self.kwargs).astype(raw.data.dtype)
            
            new_batch = gp.Batch()
            noised_spec = raw.spec.copy()
            noised = gp.Array(noised_data, noised_spec)
            new_batch[self.new_array] = noised
            
            return new_batch

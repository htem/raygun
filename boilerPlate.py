import numpy as np
import torch
import random
import gunpowder as gp

from skimage import filters

class BoilerPlate(gp.BatchFilter):
  def __init__(self, in_array, out_array):
    self.in_array = in_array
    self.out_array = out_array

  def setup(self):
    # tell downstream nodes about the new array
    self.provides(
      self.out_array,
      self.spec[self.in_array].copy())

  def prepare(self, request):
    print(f"{self.prefix}\tRequest going upstream: {request}")

  def process(self, batch, request):
    data = batch[self.in_array].data ** 2 #ex: square array data from batch
    # batch[self.array].spec holds array specs, like ROI
    batch[self.out_array].data = data # ex: put it back and pass it down the line
    # print(f"{self.prefix}\tBatch going downstream: {batch}")

class Smooth(gp.BatchFilter):

  def __init__(self, array, sigma):
    self.array = array
    self.sigma = sigma
    self.truncate = 4

  def prepare(self, request):

    # the requested ROI for array
    roi = request[self.array].roi

    # 1. compute the context
    context = gp.Coordinate((self.truncate,)*roi.dims()) * self.sigma

    # 2. enlarge the requested ROI by the context
    context_roi = roi.grow(context, context)

    # create a new request with our dependencies
    deps = gp.BatchRequest()
    deps[self.array] = context_roi

    # return the request
    return deps

  def process(self, batch, request):

    # 3. smooth the whole array (including the context)
    data = batch[self.array].data
    batch[self.array].data = filters.gaussian(
      data,
      sigma=self.sigma,
      truncate=self.truncate)

    # 4. crop the array back to the request
    batch[self.array] = batch[self.array].crop(request[self.array].roi)
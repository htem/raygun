import copy
from typing import List
import logging

import numpy as np
import gunpowder as gp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Unsqueeze(gp.BatchFilter):
    """Unsqueeze a batch at a given axis

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys to unsqueeze.
        axis: Position where the new axis is placed, defaults to 0.
    """

    def __init__(self, arrays: List[gp.ArrayKey], axis: int = 0):
        self.arrays = arrays
        self.axis = axis

        if self.axis != 0:
            raise NotImplementedError(
                'Unsqueeze only supported for leading dimension')

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = np.expand_dims(batch[array].data, self.axis)
        return outputs


class Squeeze(gp.BatchFilter):
    """Squeeze a batch at a given axis

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys to squeeze.
        axis: Position of the single-dimensional axis to remove, defaults to 0.
    """

    def __init__(self, arrays: List[gp.ArrayKey], axis: int = 0):
        self.arrays = arrays
        self.axis = axis

        if self.axis != 0:
            raise NotImplementedError(
                'Squeeze only supported for leading dimension')

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = np.squeeze(batch[array].data, self.axis)
            logger.debug(f'{array} shape: {outputs[array].data.shape}')

        return outputs

class DeepCopy(gp.BatchFilter):
    """ deep copy arrays to ensure that they are contiguous in memory

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys for arrays to be copied
    """

    def __init__(self, arrays: List[gp.ArrayKey]):
        self.arrays = arrays

    # copy the specs because everything is copied here
    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = batch[array].data.copy()
        return outputs

class ToDtype(gp.BatchFilter):
    """ Cast arrays to another numerical datatype

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys for typecasting
        dtype: output data type as string
        output_arrays (List[gp.ArrayKey]): optional, ArrayKeys for outputs
    """

    def __init__(self,
                 arrays: List[gp.ArrayKey],
                 dtype,
                 output_arrays: List[gp.ArrayKey] = None):
        self.arrays = arrays
        self.dtype = dtype

        if output_arrays:
            assert len(arrays) == len(output_arrays)
        self.output_arrays = output_arrays

    def setup(self):
        self.enable_autoskip()

        if self.output_arrays:
            for in_array, out_array in zip(self.arrays, self.output_arrays):
                if not out_array:
                    raise NotImplementedError(
                        'Provide no output_arrays or one for each input_array')
                else:
                    self.provides(out_array, self.spec[in_array].copy())
        else:
            for array in self.arrays:
                self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            deps[in_array] = request[out_array].copy()

        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            outputs[out_array] = copy.deepcopy(batch[in_array])
            logger.debug((
                f'{type(self).__name__} upstream provider spec dtype: '
                f'{outputs[in_array].spec.dtype}'
            ))
            outputs[out_array].spec.dtype = self.dtype
            outputs[out_array].data = batch[in_array].data.astype(self.dtype)

        return outputs

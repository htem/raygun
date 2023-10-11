from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
import random
import numpy as np

class SmoothArray(gp.BatchFilter):
    def __init__(self, array, blur_range):
        self.array = array
        self.range = blur_range

    def process(self, batch, request):

        array = batch[self.array].data

        assert len(array.shape) == 3

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(self.range[0], self.range[1])

        for z in range(array.shape[0]):
            array_sec = array[z]

            array[z] = np.array(
                    gaussian_filter(array_sec, sigma=sigma)
            ).astype(array_sec.dtype)

        batch[self.array].data = array


class RandomNoiseAugment(gp.BatchFilter):
    def __init__(self, array, seed=None, clip=True, **kwargs):
        self.array = array
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        mode = random.choice(["gaussian","poisson","none", "none"])

        if mode != "none":
            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Noise augmentation requires float types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
            if self.clip:
                assert raw.data.min() >= -1 and raw.data.max() <= 1, "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."
            raw.data = random_noise(
                raw.data,
                mode=mode,
                seed=self.seed,
                clip=self.clip,
                **self.kwargs).astype(raw.data.dtype)
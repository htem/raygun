# %%
from functools import partial
import daisy
from numpy import unique
import numpy as np
import skimage.morphology
import matplotlib.pyplot as plt

source_file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvBottomGT/CBxs_lobV_bottomp100um_training_0.n5'
source_ds = 'volumes/gt_CBvBottomGT_training_0_BrianReicher_20220513'
source = daisy.open_ds(source_file, source_ds)

roi = source.data_roi.grow((-480*30, -400*30, -456*30), (-480*30, -496*30, -400*30))
data = source.to_ndarray(roi)

# %%
radius = 2
area_threshold = 128
diameter_threshold = 64

footprint = skimage.morphology.ball(radius)
# ero_di = lambda x: skimage.morphology.dilation(skimage.morphology.erosion(x, footprint), footprint)
# ero_di = lambda x: skimage.morphology.dilation(skimage.morphology.erosion(x))

methods = [
        'closing', 
        # 'area_closing', 
        # 'diameter_closing', 
        # ero_di, 
        'opening'
        ]
methods_kwargs = {'closing': {'footprint': footprint},
                'opening': {'footprint': footprint},
                # 'area_closing': {'area_threshold': area_threshold},
                # 'diameter_closing': {'diameter_threshold': diameter_threshold}
                }
fig, axs = plt.subplots(1, 1+len(methods), figsize=(10*(1+len(methods)),10))
axs[0].imshow(data[data.shape[0]//2,...])
axs[0].set_title('Original')
for i, method in enumerate(methods):
    if type(method) is str:
        this = getattr(skimage.morphology, method)(data, **methods_kwargs[method])
    else:
        this = method(data)
    axs[i+1].imshow(this[data.shape[0]//2,...])
    axs[i+1].set_title(method)

this = []
# this.append(skimage.morphology.opening(skimage.morphology.dilation(data, footprint), footprint))
# this.append(skimage.morphology.closing(skimage.morphology.erosion(data, footprint), footprint))
this.append(skimage.morphology.closing(skimage.morphology.opening(data, footprint), footprint))
this.append(skimage.morphology.opening(skimage.morphology.closing(data, footprint), footprint))
# this.append(skimage.morphology.closing(skimage.morphology.closing(data, footprint), footprint))
# this.append()
# this.append()

fig, axs = plt.subplots(1, len(this), figsize=(10*len(this),10))
for i, that in enumerate(this):
    axs[i].imshow(that[data.shape[0]//2,...])

#%%

def smooth_labels(data, method, radius):
    labels = np.unique(data)
    if hasattr(skimage.morphology, method):
        footprint = skimage.morphology.ball(radius)
        part_method = partial(getattr(skimage.morphology, 'binary_' + method), footprint=footprint)
    else:
        methods = {}
        methods['close'] = partial(smooth_labels, method='closing', radius=radius)
        methods['open'] = partial(smooth_labels, method='opening', radius=radius)
        order = method.split('-')
        part_method = lambda x: methods[order[1]](methods[order[0]](x))
        # part_method = lambda x: x
        # for this_method in method.split('-'):
        #     part_method = methods[this_method](part_method)

    smoothed = np.zeros_like(data)
    for label in labels:
        mask = data == label
        smooth_mask = part_method(mask)
        smoothed[smooth_mask] = label

    return smoothed


# %%
radius = 2
footprint = skimage.morphology.ball(radius)
methods = [
        'closing', 
        'opening',
        'open-close',
        'close-open'
        ]
fig, axs = plt.subplots(1, 1+len(methods), figsize=(10*(1+len(methods)),10))
axs[0].imshow(data[data.shape[0]//2,...])
axs[0].set_title('Original')
for i, method in enumerate(methods):
    this = smooth_labels(data, method, radius)
    axs[i+1].imshow(this[data.shape[0]//2,...])
    axs[i+1].set_title(f'{method} r={radius}')

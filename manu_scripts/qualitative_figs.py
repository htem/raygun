#%%
import os
import matplotlib.pyplot as plt
import daisy
import numpy as np
from skimage.color import label2rgb

def get_image(file, ds, roi):
    dataset = daisy.open_ds(file, ds)
    if all([shape >= 2*vx_size for shape, vx_size in zip(roi.get_shape(), dataset.voxel_size)]):
        cut = (roi.get_shape()[-1] - dataset.voxel_size[-1]) / 2
        roi = roi.grow((0,0,-cut), (0,0,-cut)).snap_to_grid(dataset.voxel_size)
    return dataset.to_ndarray(roi)

def get_images(file, datasets, roi):
    images = {}
    for dataset in datasets:
        if isinstance(dataset, list):
            seg_im = label2rgb(get_image(file, dataset[0], roi))
            mask_im = get_image(file, dataset[1], roi)
            images['GT_segmentation'] = np.concatenate((seg_im, mask_im))
        else:
            images[dataset] = get_image(file, dataset, roi)
    return images

def show_images(file, datasets, roi, save=False):
    if save:
        filepath = os.path.splitext(os.path.basename(file))[0]
        os.makedirs(filepath, exist_ok =True)
    images = get_images(file, datasets, roi)
    num = len(images)
    fig, axs = plt.subplots(1, num, figsize=(20, 20*num))
    if num == 1:
        axs = [axs]
    for ax, (key, image) in zip(axs, images.items()):
        if 'segmentation' in key or len(image.shape) >= 3:
            if save:
                plt.imsave(os.path.join(filepath, key, '.png'), image, vmin=0, vmax=255)
            ax.imshow(image, title=key, vmin=0, vmax=255)
        else:
            if save:
                plt.imsave(os.path.join(filepath, key, '.png'), image, vmin=0, vmax=255, cmap='gray')
            ax.imshow(image, title=key, vmin=0, vmax=255, cmap='gray')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return fig

#%%
file = '/n/groups/htem/Segmentation/shared-nondev/cbx_fn/gt_xnh/CBxs_lobV/CBxs_lobV_bottomp100um_cutout0/gt_mk_bugfix.zarr'
roi = daisy.Roi(offset=(896, 1920, 312), shape=(400,400,400))*30
datasets = ['volumes/interpolated_90nm_aligned', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed13_checkpoint330000_netG2_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed42_checkpoint310000_netG2_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed4_checkpoint310000_netG2_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed13_checkpoint350000_netG2_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed3_checkpoint340000_netG2_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG2_184tCrp']

fig = show_images(file, datasets, roi)

#%%
file = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_1.n5'
roi = daisy.Roi(offset=(1737, 600, 1118), shape=(400,400,400))*30
datasets = ['volumes/raw_30nm', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed13_checkpoint330000_netG1_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed42_checkpoint310000_netG1_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407LinkNoBottle_seed4_checkpoint310000_netG1_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed13_checkpoint350000_netG1_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed3_checkpoint340000_netG1_184tCrp', 'volumes/CycleGun_CBxFN90nmTile2_CBv30nmBottom100um_20220407SplitNoBottle_seed42_checkpoint340000_netG1_184tCrp']

fig = show_images(file, datasets, roi)

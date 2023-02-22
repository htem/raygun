#%%
import os
import matplotlib.pyplot as plt
import daisy
import numpy as np
from skimage.color import label2rgb


def get_image(file, ds, roi):
    dataset = daisy.open_ds(file, ds)
    if all(
        [
            shape >= 2 * vx_size
            for shape, vx_size in zip(roi.get_shape(), dataset.voxel_size)
        ]
    ):
        cut = (roi.get_shape()[-1] - dataset.voxel_size[-1]) / 2
        roi = roi.grow((0, 0, -cut), (0, 0, -cut)).snap_to_grid(dataset.voxel_size)
    return dataset.to_ndarray(roi)


def get_images(dataset_dict, roi):
    images = {}
    for file, datasets in dataset_dict.items():
        for name, dataset in datasets.items():
            if isinstance(dataset, list):
                seg_im = label2rgb(get_image(file, dataset[0], roi))
                mask_im = get_image(file, dataset[1], roi)
                images["GT_segmentation"] = np.concatenate((seg_im, mask_im))
            else:
                images[name] = get_image(file, dataset, roi)
    return images


def show_images(dataset_dict, roi, save=False, axs=None):
    if save:
        filepath = os.path.splitext(os.path.basename(file))[0]
        os.makedirs(filepath, exist_ok=True)
    images = get_images(dataset_dict, roi)
    num = len(images)
    if axs is None:
        fig, axs = plt.subplots(1, num, figsize=(20, 20 * num))
    if num == 1:
        axs = [axs]
    for ax, (key, image) in zip(axs, images.items()):
        image = image.squeeze()
        if len(image.shape) >= 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose((1, 2, 0))
            if image.shape[-1] > 3:
                image = image[..., :3]

        if "segmentation" in key or len(image.shape) >= 3:
            if save:
                plt.imsave(os.path.join(filepath, key, ".png"), image, vmin=0, vmax=255)
            ax.imshow(image, vmin=0, vmax=255)
            ax.set_title(key)
        else:
            if save:
                plt.imsave(
                    os.path.join(filepath, key, ".png"),
                    image,
                    vmin=0,
                    vmax=255,
                    cmap="gray",
                )
            ax.imshow(image, vmin=0, vmax=255, cmap="gray")
            ax.set_title(key)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    # return fig, axs

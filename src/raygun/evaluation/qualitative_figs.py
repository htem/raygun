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


#%% Training0
num = 4
fig, axes = plt.subplots(2, num, figsize=(5 * num, 10))


roi = daisy.Roi(offset=(896, 1920, 312), shape=(512, 512, 1)) * 30
dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvBottomGT/training_0.n5": {
        "Real 30nm": "volumes/raw_30nm",
        "Real 90nm": "volumes/interpolated_90nm_aligned",
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/link/seed3/training_0.n5": {
        "Link: Fake 90nm (best)": "volumes/raw_30nm_netG2_56000"  # picked based on final test performance
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/split/seed42/training_0.n5": {
        "Split: Fake 90nm (best)": "volumes/raw_30nm_netG2_36000"  # picked based on final test performance
    },
}

show_images(dataset_dict, roi, axs=axes[0])

#%%
# Eval1
# roi = daisy.Roi(offset=(1737, 600, 1118), shape=(512, 512, 1)) * 30
roi = daisy.Roi(offset=(1737, 600, 1589), shape=(512, 512, 1)) * 30
dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
        "Real 30nm": "volumes/raw_30nm",
        "Real 90nm": "volumes/interpolated_90nm_aligned",
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/link/seed13/eval_1.n5": {
        "Link: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_46000"  # picked based on final test performance
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/split/seed42/eval_1.n5": {
        "Split: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_36000"  # picked based on final test performance
    },
}

show_images(dataset_dict, roi, axs=axes[1])
fig.set_tight_layout(True)
fig
# %%
# Now Segmentations
num = 4
fig, axes = plt.subplots(2, num, figsize=(5 * num, 10))

# roi = daisy.Roi(offset=(1737, 600, 1118), shape=(512, 512, 1)) * 30
roi = daisy.Roi(offset=(1737, 600, 1589), shape=(512, 512, 1)) * 30
# Trained on Real 30nm
dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
        "Raw": "volumes/raw_30nm"
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/03_evaluate_7/train_real/30nm/predict_real/30nm/eval_1.n5": {
        "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
        "Predicted\nAffinities": "pred_affs",
        "Predicted\nSegmentation": "segment",
    },
}
show_images(dataset_dict, roi, axs=axes[0])
#%%

# Trained on 90nm
dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
        "Real 30nm": "volumes/raw_30nm",
        "Real 90nm": "volumes/interpolated_90nm_aligned",
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/link/seed13/eval_1.n5": {
        "Link: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_46000"  # picked based on final test performance
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2022/01_cycleGAN_7/split/seed42/eval_1.n5": {
        "Split: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_36000"  # picked based on final test performance
    },
}

show_images(dataset_dict, roi, axs=axes[1])
fig.set_tight_layout(True)
fig

#%%
import os
import matplotlib.pyplot as plt
import daisy
import numpy as np
from skimage.color import label2rgb
import matplotlib

# switch to svg backend
matplotlib.use("svg")
# update latex preamble
plt.rcParams.update(
    {
        "svg.fonttype": "path",
        # "font.family": "sans-serif",
        # "font.sans-serif": [
        #     "Avenir",
        #     "AvenirNextLTPro",
        #     "Avenir Next LT Pro",
        #     "AvenirNextLTPro-Regular",
        #     "UniversLTStd-Light",
        #     "Verdana",
        #     "Helvetica",
        # ],
        "path.simplify": True,
        # "text.usetex": True,
        # "pgf.rcfonts": False,
        # "pgf.texsystem": 'pdflatex', # default is xetex
        # "pgf.preamble": [
        #      r"\usepackage[T1]{fontenc}",
        #      r"\usepackage{mathpazo}"
        #      ]
        # "font.size": 20,
    }
)


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
            images[name] = get_image(file, dataset, roi)
            if "segment" in dataset or "segment" in name:
                images[name] = label2rgb(images[name], bg_label=0)
    return images


def show_images(
    dataset_dict,
    roi,
    axs=None,
    save=False,
    folder_name="qualitative_figures",
    overlay_seg=None,
):
    # if save:
    #     filepath = os.path.join(os.getcwd(), folder_name)
    #     os.makedirs(filepath, exist_ok=True)
    if overlay_seg is not None:
        seg = get_image(overlay_seg["file"], overlay_seg["ds"], roi)
        seg = seg.squeeze()
        if len(seg.shape) >= 3 and seg.shape[0] < seg.shape[-1]:
            seg = seg.transpose((1, 2, 0))
            if seg.shape[-1] > 3:
                seg = seg[..., :3]
        seg = label2rgb(seg, bg_label=0)
        alpha = (seg.max(-1) > 0).astype(float)
        alpha *= 0.5
        seg = np.concatenate((seg, alpha[..., None]), axis=2)
    else:
        seg = None
    images = get_images(dataset_dict, roi)
    num = len(images)
    if axs is None:
        fig, axs = plt.subplots(1, num, figsize=(20, 20 * num))
    if num == 1:
        axs = [axs]
    for ax, (key, image) in zip(axs, images.items()):
        # name = copy(key)
        # for sep in ["\n", "(", ")", " ", ":"]:
        #     name = name.replace(sep, "_")
        image = image.squeeze()
        if len(image.shape) >= 3 and image.shape[0] < image.shape[-1]:
            image = image.transpose((1, 2, 0))
            if image.shape[-1] > 3:
                image = image[..., :3]
        images[key] = image
        # if seg is not None:
        #     image = label2rgb(seg, image=image, alpha=0.7, bg_label=0)
        if "segment" in key or len(image.shape) >= 3:
            # if save:
            #     plt.imsave(os.path.join(filepath, name, ".svg"), image, vmin=0, vmax=255)
            ax.imshow(image, vmin=0, vmax=255)
            ax.set_title(key)
        else:
            # if save:
            #     plt.imsave(
            #         os.path.join(filepath, name, ".svg"),
            #         image,
            #         vmin=0,
            #         vmax=255,
            #         cmap="gray",
            #     )
            ax.imshow(image, vmin=0, vmax=255, cmap="gray")
            ax.set_title(key)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if seg is not None:
            ax.imshow(seg)

    return images, seg

    # return fig, axs


#%% Training0
num = 4
fig, axes = plt.subplots(2, num, figsize=(5 * num, 10))

dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvBottomGT/training_0.n5": {
        "Real 30nm": "volumes/raw_30nm",
        "Real 90nm": "volumes/interpolated_90nm_aligned",
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/link/seed42/training_0.n5": {
        "Link: Fake 90nm (best)": "volumes/raw_30nm_netG2_62000"  # picked based on final test performance
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/split/seed42/training_0.n5": {
        "Split: Fake 90nm (best)": "volumes/raw_30nm_netG2_36000"  # picked based on final test performance
    },
}
# file = list(dataset_dict.keys())[0]
# ds = daisy.open_ds(file, list(dataset_dict[file].values())[0])
# roi = daisy.Roi(offset=(896, 1920, 312), shape=(512, 512, 1)) * 30
roi = daisy.Roi(offset=(884, 1908, 312), shape=(512, 512, 1)) * 30
# roi = daisy.Roi(offset=(884, 1908, 340), shape=(512, 512, 1)) * 30

images, seg = show_images(
    dataset_dict,
    roi,
    axs=axes[0],
    overlay_seg={
        "file": "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvBottomGT/training_0.n5",
        "ds": "volumes/GT_labels",
    },
)
fig

#%%
# Eval1
# roi = daisy.Roi(offset=(1737, 600, 1118), shape=(512, 512, 1)) * 30
roi = daisy.Roi(offset=(1737, 600, 1589), shape=(512, 512, 1)) * 30
dataset_dict = {
    "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
        "Real 30nm": "volumes/raw_30nm",
        "Real 90nm": "volumes/interpolated_90nm_aligned",
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/link/seed13/eval_1.n5": {
        "Link: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_46000"  # picked based on final test performance
    },
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/split/seed42/eval_1.n5": {
        "Split: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_36000"  # picked based on final test performance
    },
}

images, seg = show_images(dataset_dict, roi, axs=axes[1])
fig.set_tight_layout(True)
fig
# %%
# Now Segmentations
# roi = daisy.Roi(offset=(1737, 600, 1118), shape=(512, 512, 1)) * 30
roi = daisy.Roi(offset=(1737, 600, 1589), shape=(512, 512, 1)) * 30
# Trained on Real 30nm
dataset_dict_list = [
    {  # Train on Real 30nm, Predict on Real 30nm
        "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
            "Real 30nm": "volumes/raw_30nm"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_real/30nm/predict_real/30nm/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Real 30nm, Predict on Link-Fake 30nm
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/link/seed13/eval_1.n5": {
            "Link: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_46000"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_real/30nm/predict_link/seed13/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Real 30nm, Predict on Split-Fake 30nm
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/01_cycleGAN/split/seed42/eval_1.n5": {
            "Split: Fake 30nm (best)": "volumes/interpolated_90nm_aligned_netG1_36000"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_real/30nm/predict_split/seed42/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Real 30nm, Predict on Real 90nm
        "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
            "Real 90nm\n(Trained on:\nreal 30nm)": "volumes/interpolated_90nm_aligned"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_real/30nm/predict_real/90nm/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Real 90nm, Predict on Real 90nm
        "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
            "Real 90nm\n(Trained on:\nreal 90nm)": "volumes/interpolated_90nm_aligned"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_real/90nm/predict_real90nm/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Link-Fake 90nm, Predict on Real 90nm
        "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
            "Real 90nm\n(Trained on:\nLink-Fake 90nm)": "volumes/interpolated_90nm_aligned"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_link/seed42/predict_real90nm/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
    {  # Train on Split-Fake 90nm, Predict on Real 90nm
        "/nrs/funke/rhoadesj/data/XNH/CBv/GT/CBvTopGT/eval_1.n5": {
            "Real 90nm\n(Trained on:\nSplit-Fake 90nm)": "volumes/interpolated_90nm_aligned"
        },
        "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/train_split/seed42/predict_real90nm/eval_1.n5": {
            "Predicted\nLocal Shape\nDescriptors": "pred_lsds",
            "Predicted\nAffinities": "pred_affs",
            "Predicted\nSegmentation": "segment",
        },
    },
]
cols = 4
rows = 7
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
for r in range(rows):
    dataset_dict = dataset_dict_list[r]
    images, seg = show_images(dataset_dict, roi, axs=axes[r])
fig.set_tight_layout(True)
fig
# %%
# Now in presentation layout
cols = 7
rows = 4
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
for c in range(cols):
    dataset_dict = dataset_dict_list[c]
    images, seg = show_images(dataset_dict, roi, axs=axes[:, c])
fig.set_tight_layout(True)
fig
# %%

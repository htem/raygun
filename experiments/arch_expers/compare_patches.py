#%%
import os
import torch
import numpy as np
import skimage
import daisy
import matplotlib.pyplot as plt

from raygun import load_system

# %%
def try_patch(system, real, side="A", pad=0, mode="eval"):
    real = np.expand_dims(real, axis=0)

    if side.upper() == "A":
        net = system.model.netG1.to("cuda")
    else:
        net = system.model.netG2.to("cuda")

    if mode.lower() == "eval":
        # system.model.eval()
        net.eval()
    elif mode.lower() == "real":
        return real, []
    else:
        # system.model.train()
        net.train()

    mid = real.shape[-1] // 2
    if pad is True:
        test = net(torch.cuda.FloatTensor(real).unsqueeze(0))
        pad = (real.shape[-1] - test.shape[-1]) // 2
        valid = True  # TODO: This is super hacky :/
    else:
        valid = False

    patch1 = torch.cuda.FloatTensor(real[:, : mid + pad, : mid + pad]).unsqueeze(0)
    patch2 = torch.cuda.FloatTensor(real[:, mid - pad :, : mid + pad]).unsqueeze(0)
    patch3 = torch.cuda.FloatTensor(real[:, : mid + pad, mid - pad :]).unsqueeze(0)
    patch4 = torch.cuda.FloatTensor(real[:, mid - pad :, mid - pad :]).unsqueeze(0)

    patches = [patch1, patch2, patch3, patch4]
    fakes = []
    for patch in patches:
        test = net(patch)
        fakes.append(test.detach().cpu().squeeze())

    if pad != 0 and not valid:
        fake_comb = torch.cat(
            (
                torch.cat((fakes[0][:-pad, :-pad], fakes[1][pad:, :-pad])),
                torch.cat((fakes[2][:-pad, pad:], fakes[3][pad:, pad:])),
            ),
            axis=1,
        )
    else:
        fake_comb = torch.cat(
            (torch.cat((fakes[0], fakes[1])), torch.cat((fakes[2], fakes[3]))), axis=1
        )

    return fake_comb.squeeze(), real.squeeze()


def get_center_roi(ds, side_length, ndims, voxel_size=None):
    if voxel_size is None:
        voxel_size = ds.voxel_size
    shape = list((1,) * 3)
    shape[:ndims] = (side_length * 2,) * ndims
    real_shape = voxel_size * daisy.Coordinate(shape)
    roi = daisy.Roi(ds.roi.center, real_shape)
    roi = roi.shift(-real_shape / 2).snap_to_grid(ds.voxel_size)
    return roi


def get_real(system, side, roi=None):
    ds = daisy.open_ds(system.sources[side]["path"], system.sources[side]["real_name"])
    if roi is None:
        roi = get_center_roi(
            ds,
            system.side_length,
            system.ndims,
            daisy.Coordinate(system.common_voxel_size),
        )

    data = ds.to_ndarray(roi)
    data = data / 255.0
    if ds.voxel_size != daisy.Coordinate(system.common_voxel_size):
        data = skimage.transform.rescale(
            data,
            np.array(ds.voxel_size) / np.array(system.common_voxel_size),
        )

    if len(data.shape) > system.ndims:
        data = data[..., 0]
    return data


def show_patches(system, pad=0, checkpoint=None):
    if isinstance(system, str):
        system = load_system(system)
        system.build_system()
        system.load_saved_model(checkpoint=checkpoint)

    fig, axs = plt.subplots(2, 3, figsize=(30, 20))
    reals = []
    # print(f"{system.model_name} at iteration {system.iteration}")
    for i, side in enumerate(["A", "B"]):
        axs[i, 0].set_ylabel(side)
        real = get_real(system, side)
        reals.append(real)
        for j, mode in enumerate(["real", "eval", "train"]):
            if mode == "real":
                img = real
            else:
                img, _ = try_patch(
                    system,
                    side=side,
                    mode=mode,
                    pad=pad,
                    real=real,
                )
            axs[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axs[i, j].set_title(mode)
    return system, fig, real


#%%
if __name__ == "__main__":
    train_confs = [
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms/allTrain/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms/noNorm/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms/noTrack/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms/switch/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms_split/allTrain/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms_split/noNorm/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms_split/noTrack/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/resnet_2D_norms_split/switch/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D/link/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D/link_LR/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D/split/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D/split_LR/train_conf.json",
        "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D_512/link/train_conf.json",
        "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D_512/link_LR/train_conf.json",
        "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D_512/split/train_conf.json",
        # "/nrs/funke/rhoadesj/raygun/experiments/arch_expers/unet_2D_512/split_LR/train_conf.json",
    ]

    for train_conf in train_confs:
        os.chdir(os.path.dirname(train_conf))

        system, fig, real = show_patches(
            train_conf,
            pad=16,
        )

        fig.savefig(
            f"{os.path.dirname(train_conf).split('/')[-1]}_patch_compare.png",
            bbox_inches="tight",
        )

# %%

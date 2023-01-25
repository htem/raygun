#%%
import os
import sys
import daisy
import torch
import numpy as np
from raygun import load_system, read_config
import matplotlib.pyplot as plt

#%%
render_config_path = "/nrs/funke/rhoadesj/raygun/experiments/test_runs/MTLSD/30nm_eroded/predict_training0.json"
render_config = {  # Defaults
    "crop": 0,
    "read_size": None,
    "max_retries": 2,
    "num_workers": 16,
    "ndims": None,
    "net_name": None,
    "scaleShift_input": None,
    "output_ds": None,
    "out_specs": None,
}

temp = read_config(render_config_path)
render_config.update(temp)

config_path = render_config["config_path"]
train_config = read_config(config_path)
source_path = render_config["source_path"]
source_dataset = render_config["source_dataset"]
net_name = render_config["net_name"]
checkpoint = render_config["checkpoint"]
output_ds = render_config["output_ds"]
scaleShift_input = render_config["scaleShift_input"]
crop = render_config["crop"]
ndims = render_config["ndims"]
if ndims is None:
    ndims = train_config["ndims"]
#%%
system = load_system(config_path)
if not os.path.exists(str(checkpoint)):
    checkpoint_path = os.path.join(
        os.path.dirname(config_path),
        system.checkpoint_basename.lstrip("./") + f"_checkpoint_{checkpoint}",
    )

    if not os.path.exists(checkpoint_path):
        checkpoint_path = None

else:
    checkpoint_path = None

system.load_saved_model(checkpoint_path)
if net_name is not None:
    model = getattr(system.model, net_name)
else:
    model = system.model

model.eval()
#%%
source = daisy.open_ds(source_path, source_dataset)
gt = daisy.open_ds(source_path, train_config["sources"][0]["labels"])
if "input_shape" in render_config.keys() or "input_shape" in train_config.keys():
    try:
        input_shape = render_config["input_shape"]
        output_shape = render_config["output_shape"]
    except:
        input_shape = train_config["input_shape"]
        output_shape = train_config["output_shape"]

    if not isinstance(input_shape, list):
        input_shape = daisy.Coordinate((1,) * (3 - ndims) + (input_shape,) * (ndims))
        output_shape = daisy.Coordinate((1,) * (3 - ndims) + (output_shape,) * (ndims))
    else:
        input_shape = daisy.Coordinate(input_shape)
        output_shape = daisy.Coordinate(output_shape)

    read_size = input_shape * source.voxel_size
    write_size = output_shape * source.voxel_size
    context = (read_size - write_size) // 2
    read_roi = daisy.Roi((0, 0, 0), read_size)
    write_roi = daisy.Roi(context, write_size)

else:
    read_size = render_config["read_size"]  # CHANGE TO input_shape
    if read_size is None:
        read_size = train_config["side_length"]  # CHANGE TO input_shape
    crop = render_config["crop"]
    read_size = daisy.Coordinate((1,) * (3 - ndims) + (read_size,) * (ndims))
    crop = daisy.Coordinate((0,) * (3 - ndims) + (crop,) * (ndims))

    read_roi = daisy.Roi([0, 0, 0], source.voxel_size * read_size)
    write_size = read_size - crop * 2
    write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)

this_roi = read_roi.shift(source.roi.center)
data = source.to_ndarray(this_roi)
gt_data = gt.to_ndarray(this_roi)
#%%
data = source.to_ndarray(this_roi)
# ind = data.shape[0]//2
ind = 57
data_img = data[ind]
gt_img = gt_data[ind]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(data_img, cmap="gray")
axs[1].imshow(gt_img)

#%%
data = source.to_ndarray(this_roi)
data = torch.FloatTensor(data).unsqueeze(0)
if ndims == 3:
    data = data.unsqueeze(0)

data -= np.iinfo(source.dtype).min  # TODO: Assumes integer inputs
data /= np.iinfo(source.dtype).max

if scaleShift_input is not None:
    data *= scaleShift_input[0]
    data += scaleShift_input[1]

outs = model(data)
#%%
# ind = out_data.shape[2]//2
ind = 10
out_data = outs[0].detach().squeeze().numpy().transpose((1, 2, 3, 0))
data = source.to_ndarray(this_roi)
pad = int(np.subtract(data.shape[0], out_data.shape[0]) // 2)
data_img = data[pad:-pad, pad:-pad, pad:-pad]
data_img = data_img[ind]
gt_img = gt_data[pad:-pad, pad:-pad, pad:-pad]
gt_img = gt_img[ind]
out_img = out_data[ind, ..., :3]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(data_img, cmap="gray")
axs[1].imshow(gt_img)
axs[2].imshow(out_img)
out_data.max()
#%%

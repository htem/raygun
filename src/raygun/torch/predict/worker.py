import os
import sys
import daisy
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

from raygun import load_system, read_config


def worker(render_config_path):
    client = daisy.Client()
    worker_id = client.worker_id
    logger = logging.getLogger(f"crop_worker_{worker_id}")
    logger.info(f"Launching {worker_id}...")

    # Setup rendering pipeline
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
    if torch.cuda.is_available():
        logger.info("Moving model to CUDA...")
        model.to("cuda")  # TODO pick best GPU

    del system

    source = daisy.open_ds(source_path, source_dataset)

    # Load output datsets
    dest_path = os.path.join(
        os.path.dirname(config_path), os.path.basename(source_path)
    )

    if output_ds is None:
        if net_name is not None:
            output_ds = [f"{source_dataset}_{net_name}_{checkpoint}"]
        else:
            output_ds = [f"{source_dataset}_{checkpoint}"]
    destinations = {}
    for dest_dataset in output_ds:
        destinations[dest_dataset] = daisy.open_ds(dest_path, dest_dataset, "a")

    while True:
        with client.acquire_block() as block:
            if block is None:
                break

            else:
                data = source.to_ndarray(block.read_roi)
                if torch.cuda.is_available():
                    data = torch.cuda.FloatTensor(data).unsqueeze(0)
                else:
                    data = torch.FloatTensor(data).unsqueeze(0)

                if ndims == 3:
                    data = data.unsqueeze(0)

                data -= np.iinfo(source.dtype).min  # TODO: Assumes integer inputs
                data /= np.iinfo(source.dtype).max

                if scaleShift_input is not None:
                    data *= scaleShift_input[0]
                    data += scaleShift_input[1]

                outs = model(data)
                del data

                if not isinstance(outs, tuple):
                    outs = tuple([outs])

                for out, dest_dataset in zip(outs, output_ds):
                    destination = destinations[dest_dataset]
                    out = out.detach().squeeze()
                    if crop and crop != 0:
                        if ndims == 2:
                            out = out[crop:-crop, crop:-crop]
                        elif ndims == 3:
                            out = out[crop:-crop, crop:-crop, crop:-crop]
                        else:
                            raise NotImplementedError()

                    try:
                        out *= np.iinfo(destination.dtype).max
                        out = torch.clamp(
                            out,
                            np.iinfo(destination.dtype).min,
                            np.iinfo(destination.dtype).max,
                        )
                    except:
                        logger.info(
                            f"Assuming output data for {dest_dataset} is float between 0 and 1..."
                        )
                        out = torch.clamp(out, 0, 1)

                    if torch.cuda.is_available():
                        out = out.cpu().numpy().astype(destination.dtype)
                    else:
                        out = out.numpy().astype(destination.dtype)

                    if (
                        ndims == 2 and len(out.shape) < 3
                    ):  # Add Z dimension if necessary
                        out = out[None, ...]
                    elif (
                        ndims == 2 and len(out.shape) == 3
                    ):  # Add Z dimension if necessary
                        out = out[:, None, ...]

                    # if len(out.shape) < 4:  # Add channel dimension if necessary
                    #     out = out[None, ...]

                    destination[block.write_roi] = out
                    logger.info(f"Wrote chunk {block.block_id} to {dest_dataset}...")
                    del out


if __name__ == "__main__":
    worker(sys.argv[1])

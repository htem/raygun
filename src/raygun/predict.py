#%%
from glob import glob
from importlib import import_module
import os
from subprocess import Popen
import sys
import tempfile
import daisy
import logging

logging.basicConfig(level=logging.INFO)

from raygun import load_system, read_config

#%%
def predict(render_config_path=None):  # Use absolute path
    """Predict system (available through CLI as raygun-predict)

    Args:
        config_path (str, optional): Path to json file for predicting configuration. Defaults to command line argument.
    """
    if render_config_path is None:
        render_config_path = sys.argv[1]

    logger = logging.getLogger(__name__)
    logger.info("Loading prediction config...")

    render_config = {  # Defaults
        "crop": 0,
        "read_size": None,
        "max_retries": 2,
        "num_workers": 16,
        "ndims": None,
        "net_name": None,
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
    # compressor = render_config['compressor']
    num_workers = render_config["num_workers"]
    max_retries = render_config["max_retries"]
    output_ds = render_config["output_ds"]
    out_specs = render_config["out_specs"]
    ndims = render_config["ndims"]
    if ndims is None:
        ndims = train_config["ndims"]

    dest_path = os.path.join(
        os.path.dirname(config_path), os.path.basename(source_path)
    )
    if output_ds is None:
        if net_name is not None:
            output_ds = [f"{source_dataset}_{net_name}_{checkpoint}"]
        else:
            output_ds = [f"{source_dataset}_{checkpoint}"]

    source = daisy.open_ds(source_path, source_dataset)

    # Get input/output sizes #TODO: Clean this up with refactor prior to 0.3.0 (will break old CGAN configs...)
    if "input_shape" in render_config.keys() or "input_shape" in train_config.keys():
        try:
            input_shape = render_config["input_shape"]
            output_shape = render_config["output_shape"]
        except:
            input_shape = train_config["input_shape"]
            output_shape = train_config["output_shape"]

        if not isinstance(input_shape, list):
            input_shape = daisy.Coordinate(
                (1,) * (3 - ndims) + (input_shape,) * (ndims)
            )
            output_shape = daisy.Coordinate(
                (1,) * (3 - ndims) + (output_shape,) * (ndims)
            )
        else:
            input_shape = daisy.Coordinate(input_shape)
            output_shape = daisy.Coordinate(output_shape)

        read_size = input_shape * source.voxel_size
        write_size = output_shape * source.voxel_size
        context = (read_size - write_size) // 2
        read_roi = daisy.Roi((0, 0, 0), read_size)
        write_roi = daisy.Roi(context, write_size)

    else:
        read_size = render_config["read_size"]  # TODO: CHANGE TO input_shape
        if read_size is None:
            read_size = train_config["side_length"]  # TODO: CHANGE TO input_shape
        crop = render_config["crop"]
        read_size = daisy.Coordinate((1,) * (3 - ndims) + (read_size,) * (ndims))
        crop = daisy.Coordinate((0,) * (3 - ndims) + (crop,) * (ndims))

        read_roi = daisy.Roi([0, 0, 0], source.voxel_size * read_size)
        write_size = read_size - crop * 2
        write_roi = daisy.Roi(source.voxel_size * crop, source.voxel_size * write_size)

    # Prepare output datasets
    for dest_dataset in output_ds:
        these_specs = {
            "filename": dest_path,
            "ds_name": dest_dataset,
            "total_roi": source.data_roi,
            "voxel_size": source.voxel_size,
            "dtype": source.dtype,
            "write_size": write_roi.get_shape(),
            "num_channels": None,
            "delete": True,
        }
        if out_specs is not None and dest_dataset in out_specs.keys():
            these_specs.update(out_specs[dest_dataset])

        destination = daisy.prepare_ds(**these_specs)

    # Make temporary directory for storing log files
    with tempfile.TemporaryDirectory() as temp_dir:
        cur_dir = os.getcwd()
        os.chdir(temp_dir)
        print(f"Executing in {os.getcwd()}")

        if "launch_command" in render_config.keys():
            launch_command = [
                *render_config["launch_command"].split(" "),
                "python",
                import_module(
                    ".".join(["raygun", train_config["framework"], "predict", "worker"])
                ).__file__,
                render_config_path,
            ]
            process_function = lambda: Popen(launch_command)
            logger.info(f"Launch command: {' '.join(launch_command)}")

        else:
            worker = getattr(
                import_module(
                    ".".join(["raygun", train_config["framework"], "predict", "worker"])
                ),
                "worker",
            )
            process_function = lambda: worker(render_config_path)

        task = daisy.Task(
            os.path.basename(render_config_path).rstrip(".json"),
            total_roi=source.data_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            read_write_conflict=True,
            # fit="shrink",
            num_workers=num_workers,
            max_retries=max_retries,
            process_function=process_function,
        )

        logger.info("Running blockwise prediction...")
        if daisy.run_blockwise([task]):
            logger.info("Daisy done.")
        else:
            raise ValueError("Daisy failed.")

        logger.info("Saving viewer script...")
        view_script = os.path.join(
            os.path.dirname(config_path),
            f"view_{os.path.basename(source_path).rstrip('.n5').rstrip('.zarr')}.ng",
        )

        # Add each datasets to viewing file
        for dest_dataset in output_ds:
            if not os.path.exists(view_script):
                with open(view_script, "w") as f:
                    f.write(
                        f"neuroglancer -f {source_path} -d {source_dataset} -f {dest_path} -d {dest_dataset} "
                    )

            else:
                with open(view_script, "a") as f:
                    f.write(f"{dest_dataset} ")

        logger.info("Done.")

    os.chdir(cur_dir)


# TODO:
def _batch_predict(folder):
    os.chdir(folder)

    subfolders = glob("*/")
    if len(subfolders) > 0:
        config_paths = []
        for subfolder in subfolders:
            config_paths += _batch_predict(subfolder)

    else:
        config_path = os.path.realpath("train_conf.json")
        if os.path.exists(config_path):
            predict(config_path)
            config_paths = [config_path]

        else:
            config_paths = []

    os.chdir("..")
    return config_paths


# TODO:
def batch_predict(base_folder=None):
    """Batch predict systems (available through CLI as raygun-predict-batch).

    Args:
        base_folder (str, optional): Path to folder containing base predicting configuration json file and nested folders for batch predicting. Defaults to command line argument.
    """

    if base_folder is None:
        base_folder = sys.argv[1]

    os.chdir(base_folder)
    base_folder = os.getcwd()  # get absolute path

    config_paths = _batch_predict(".")

    os.makedirs(os.path.join(base_folder, "tensorboards"), exist_ok=True)
    while len(config_paths) > 0:
        sleep(5)
        for config_path in config_paths:
            config_name = get_config_name(config_path, base_folder)

            if not os.path.islink(
                os.path.join(base_folder, "tensorboards", config_name)
            ):
                try:
                    os.symlink(
                        os.path.join(os.path.dirname(config_path), "tensorboard"),
                        os.path.join(base_folder, "tensorboards", config_name),
                        target_is_directory=True,
                    )
                    config_paths.remove(config_path)
                except:
                    logger.info(f"Waiting for {config_name} log...")

    call("wait")


# %%
if __name__ == "__main__":
    predict(sys.argv[1])

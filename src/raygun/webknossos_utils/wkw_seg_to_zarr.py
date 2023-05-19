#%%
import argparse
import webknossos as wk
import wkw
from time import gmtime, strftime
import zipfile
import daisy
import tempfile
from glob import glob
import os


def download_wk_skeleton(
    annotation_ID,
    save_path,
    wk_url="http://catmaid2.hms.harvard.edu:9000",
    wk_token="Q9OpWh1PPwHYfH9BsnoM2Q",
    overwrite=None,
    zip_suffix=None,
):
    print(f"Downloading {wk_url}/annotations/Explorational/{annotation_ID}...")
    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(
            annotation_ID, annotation_type="Explorational"
        )

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        if overwrite is None:
            overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite is True or overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            if zip_suffix is None:
                zip_suffix = (
                    f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
                )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)
    return zip_path


def get_wk_mask(
    annotation_ID,
    save_path,  # TODO: Add mkdtemp() as default
    zarr_path,
    raw_name,
    wk_url="http://catmaid2.hms.harvard.edu:9000",
    wk_token="Q9OpWh1PPwHYfH9BsnoM2Q",
    save_name=None,
    mask_out=True,
):
    print(f"Downloading {wk_url}/annotations/Explorational/{annotation_ID}...")
    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(
            annotation_ID, annotation_type="Explorational"
        )

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            zip_suffix = input(
                f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
            )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)

    # Extract zip file
    zf = zipfile.ZipFile(zip_path)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        zipped_datafile = glob(tempdir + "/data*.zip")[0]
        print(f"Opening {zipped_datafile}...")
        zf_data = zipfile.ZipFile(zipped_datafile)
        with tempfile.TemporaryDirectory() as zf_data_tmpdir:
            zf_data.extractall(zf_data_tmpdir)

            # Open the WKW dataset (as the `1` folder)
            print(f"Opening {zf_data_tmpdir + '/1'}...")
            dataset = wkw.Dataset.open(zf_data_tmpdir + "/Volume/1")
            zarr_path = zarr_path.rstrip(os.sep)
            print(f"Opening {zarr_path}/{raw_name}...")
            ds = daisy.open_ds(zarr_path, raw_name)
            data = dataset.read(
                ds.roi.get_offset() / ds.voxel_size, ds.roi.get_shape() / ds.voxel_size
            ).squeeze()

    if save_name is not None:
        print("Saving...")
        target_roi = ds.roi
        if mask_out:
            mask_array = daisy.Array(data == 0, ds.roi, ds.voxel_size)
        else:
            mask_array = daisy.Array(data > 0, ds.roi, ds.voxel_size)

        chunk_size = ds.chunk_shape[0]
        num_channels = 1
        compressor = {
            "id": "blosc",
            "clevel": 3,
            "cname": "blosclz",
            "blocksize": chunk_size,
        }
        num_workers = 30
        write_size = mask_array.voxel_size * chunk_size
        chunk_roi = daisy.Roi(
            [
                0,
            ]
            * len(target_roi.get_offset()),
            write_size,
        )

        destination = daisy.prepare_ds(
            zarr_path,
            save_name,
            target_roi,
            mask_array.voxel_size,
            bool,
            write_size=write_size,
            write_roi=chunk_roi,
            # num_channels=num_channels,
            compressor=compressor,
        )

        # Prepare saving function/variables
        def save_chunk(block: daisy.Roi):
            # try:
            destination.__setitem__(
                block.write_roi, mask_array.__getitem__(block.read_roi)
            )
            #     return 0 # success
            # except:
            #     return 1 # error

        # Write data to new dataset
        task = daisy.Task(
            f"save>{save_name}",
            target_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit="shrink",
            num_workers=num_workers,
            max_retries=2,
        )
        success = daisy.run_blockwise([task])

        if success:
            print(
                f"{target_roi} from {annotation_name} written to {zarr_path}/{save_name}"
            )
            return destination
        else:
            print("Failed to save annotation layer.")

    else:
        if mask_out:
            return daisy.Array(data == 0, ds.roi, ds.voxel_size)
        else:
            return daisy.Array(data > 0, ds.roi, ds.voxel_size)


# Extracts and saves volume annotations as a uint32 layer alongside the zarr used for making GT (>> assumes same ROI)
def wkw_seg_to_zarr(
    annotation_ID,
    save_path,  # TODO: Add mkdtemp() as default
    zarr_path,
    raw_name,
    wk_url="http://catmaid2.hms.harvard.edu:9000",
    wk_token="Q9OpWh1PPwHYfH9BsnoM2Q",
    gt_name=None,
    gt_name_prefix="volumes/",
    overwrite=None,
):
    print(f"Downloading {annotation_ID} from {wk_url}...")
    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(
            annotation_ID#, annotation_type="Explorational"
        )

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        if overwrite is None:
            overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            zip_suffix = input(
                f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
            )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)

    # Extract zip file
    zf = zipfile.ZipFile(zip_path)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        zipped_datafile = glob(tempdir + "/data*.zip")[0]
        print(f"Opening {zipped_datafile}...")
        zf_data = zipfile.ZipFile(zipped_datafile)
        with tempfile.TemporaryDirectory() as zf_data_tmpdir:
            zf_data.extractall(zf_data_tmpdir)

            # Open the WKW dataset (as the `1` folder)
            print(f"Opening {zf_data_tmpdir + '/1'}...")
            dataset = wkw.Dataset.open(zf_data_tmpdir + "/1")
            zarr_path = zarr_path.rstrip(os.sep)
            print(f"Opening {zarr_path}/{raw_name}...")
            ds = daisy.open_ds(zarr_path, raw_name)
            data = dataset.read(
                ds.roi.get_offset() / ds.voxel_size, ds.roi.get_shape() / ds.voxel_size
            ).squeeze()
            print(f"Sum of all data: {data.sum()}")
            # Save annotations to zarr
            if gt_name is None:
                gt_name = f'{gt_name_prefix}gt_{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'

            target_roi = ds.roi
            gt_array = daisy.Array(data, ds.roi, ds.voxel_size)

            chunk_size = ds.chunk_shape[0]
            num_channels = 1
            compressor = {
                "id": "blosc",
                "clevel": 3,
                "cname": "blosclz",
                "blocksize": chunk_size,
            }
            num_workers = 30
            write_size = gt_array.voxel_size * chunk_size
            chunk_roi = daisy.Roi(
                [
                    0,
                ]
                * len(target_roi.get_offset()),
                write_size,
            )

            destination = daisy.prepare_ds(
                zarr_path,
                gt_name,
                target_roi,
                gt_array.voxel_size,
                data.dtype,
                write_size=write_size,
                # write_roi=chunk_roi,
                # num_channels=num_channels,
                # compressor=compressor,
            )

            # Prepare saving function/variables
            def save_chunk(block: daisy.Roi):
                try:
                    destination.__setitem__(
                        block.write_roi, gt_array.__getitem__(block.read_roi)
                    )
                    # destination[block.write_roi] = gt_array[block.read_roi]
                    return 0 # success
                except:
                    return 1 # error

            # Write data to new dataset
            task = daisy.Task(
                f"save>{gt_name}",
                target_roi,
                chunk_roi,
                chunk_roi,
                process_function=save_chunk,
                read_write_conflict=False,
                fit="shrink",
                num_workers=num_workers,
                max_retries=2,
            )
            success = daisy.run_blockwise([task])

            if success:
                print(
                    f"{target_roi} from {annotation_name} written to {zarr_path}/{gt_name}"
                )
                return gt_name
            else:
                print("Failed to save annotation layer.")


#%%
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "annotation_ID",
        type=str,
        help="Input wkw annotation ID (e.g. 626fd7b6010000a80079d0d9)",
    )
    ap.add_argument(
        "save_path", type=str, help="Input path to archive wkw annotation zipfile."
    )
    ap.add_argument(
        "zarr_path", type=str, help="Input path to Zarr volume used for annotation."
    )
    ap.add_argument(
        "--raw_name",
        type=str,
        help="Input name of dataset in the Zarr volume that was used for annotation.",
        default="volumes/raw"
    )
    ap.add_argument(
        "--wk_url",
        type=str,
        help="URL of Webknossos instance.",
        default="http://catmaid2.hms.harvard.edu:9000",
    )
    ap.add_argument(
        "--wk_token",
        type=str,
        help="Authentification token for Webknossos instance.",
        default="Q9OpWh1PPwHYfH9BsnoM2Q",
    )
    ap.add_argument(
        "--gt_name_prefix",
        type=str,
        help="Prefix for saving annotation layer to Zarr.",
        default="volumes/",
    )
    config = ap.parse_args()

    wkw_seg_to_zarr(**vars(config))

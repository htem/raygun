#%%
import sys
import daisy


def mask_seg(
    seg_file,
    seg_ds,
    mask_ds,
    out_ds=None,
    mask_file=None,
    out_file=None,
    mask_out=False,
):
    if mask_file is None:
        mask_file = seg_file

    if out_file is None:
        out_file = seg_file

    if out_ds is None:
        out_ds = f"{seg_ds}_masked"

    print("Loading...")
    seg = daisy.open_ds(seg_file, seg_ds)
    mask = daisy.open_ds(mask_file, mask_ds)

    print("Saving...")
    target_roi = mask.roi

    chunk_size = 64  # seg.chunk_shape[0]
    compressor = {
        "id": "blosc",
        "clevel": 3,
        "cname": "blosclz",
        "blocksize": chunk_size,
    }
    num_workers = 30
    write_size = seg.voxel_size * chunk_size
    chunk_roi = daisy.Roi(
        [
            0,
        ]
        * len(target_roi.get_offset()),
        write_size,
    )

    out = daisy.prepare_ds(
        out_file,
        out_ds,
        target_roi,
        seg.voxel_size,
        seg.dtype,
        write_size=write_size,
        write_roi=chunk_roi,
        compressor=compressor,
    )

    # Prepare saving function/variables
    def save_chunk(block: daisy.Roi):
        seg_data = seg.to_ndarray(block.read_roi)
        if mask_out:
            out_data = seg_data * (mask.to_ndarray(block.read_roi) == 0)
        else:
            out_data = seg_data * (mask.to_ndarray(block.read_roi) > 0)

        out.__setitem__(block.write_roi, out_data)

    # Write data to new dataset
    task = daisy.Task(
        f"save-{out_ds}",
        target_roi,
        read_roi=chunk_roi,
        write_roi=chunk_roi,
        process_function=save_chunk,
        read_write_conflict=False,
        fit="shrink",
        num_workers=num_workers,
        max_retries=2,
    )
    success = daisy.run_blockwise([task])

    if success:
        print(
            f"{target_roi} from {seg_ds} masked with {mask_ds} and written to {out_file}/{out_ds}"
        )
        return out
    else:
        print("Failed to save masked segmentation.")


#%%
if __name__ == "__main__":
    kwargs = {}
    keys = [
        "seg_file",
        "seg_ds",
        "mask_ds",
        "out_ds",
        "mask_file",
        "out_file",
        "mask_out",
    ]
    for i, arg in enumerate(sys.argv[1:]):
        kwargs[keys[i]] = arg
        print(f"{keys[i]}: {arg}")

    out = mask_seg(**kwargs)
# %%

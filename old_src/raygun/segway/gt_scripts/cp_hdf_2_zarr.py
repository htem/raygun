# import zarr
# import h5py
import sys
import daisy

if len(sys.argv) == 1:
    # print("Usage: python cp_zarr_2_hdf.py [raw_zarr]/[raw_ds] [segment_zarr]/[segment_ds] output.hdf")
    exit(1)

elif len(sys.argv) == 2:
    import gt_tools
    config = gt_tools.load_config(sys.argv[1], no_db=True)
    segment_f = config["segment_file"]
    h5_f = segment_f.rsplit('.', 1)[0] + '.h5'
    segment_ds_out = config["segment_ds_paintera_out"]
    segment_ds_in = config["segment_ds_paintera_in"]

else:
    raise RuntimeError("Unimplemented")

# h5_f = h5py.File(h5_f, 'r')
# h5_ds = h5_f['volumes/labels/neuron_ids']

# segment_f = sys.argv[2].split('.zarr/')[0] + '.zarr'
# segment_ds = sys.argv[2].split('.zarr/')[1]
# segment_zarr = zarr.open(segment_f, 'r+')

segment_ds_in = daisy.open_ds(segment_f, segment_ds_in)

out_ds = daisy.prepare_ds(
    segment_f,
    segment_ds_out,
    segment_ds_in.roi,
    segment_ds_in.voxel_size,
    segment_ds_in.dtype,
    # write_roi=daisy.Roi((0, 0, 0), chunk_size),
    # write_size=chunk_size,
    compressor={'id': 'zlib', 'level': 5},
    # delete=True,
    )

h5_ds = daisy.open_ds(h5_f, 'volumes/labels/neuron_ids')

print("Writing...")
out_ds[out_ds.roi] = h5_ds.to_ndarray()

# if True:
#     ds = 'volumes/raw'
#     print("Copying over dataset: %s"%ds)
#     hf.create_dataset(ds,data=raw_zarr[raw_zarr_ds][:])
#     hf[ds].attrs['resolution'] = raw_zarr[raw_zarr_ds].attrs['resolution']
#     hf[ds].attrs['offset'] = raw_zarr[raw_zarr_ds].attrs['offset']

# if True:
#     ds = 'volumes/labels/neuron_ids'
#     print("Copying over dataset: %s"%ds)
#     hf.create_dataset(ds,data=segment_zarr[segment_ds][:])
#     hf[ds].attrs['resolution'] = segment_zarr[segment_ds].attrs['resolution']
#     hf[ds].attrs['offset'] = segment_zarr[segment_ds].attrs['offset']

# hf.close()

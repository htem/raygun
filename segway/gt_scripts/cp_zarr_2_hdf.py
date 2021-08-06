# import zarr
import h5py
import sys
import daisy
import os


if len(sys.argv) == 1:
    print("Usage: python cp_zarr_2_hdf.py [raw_zarr]/[raw_ds] [segment_zarr]/[segment_ds] [output.hdf]")
    print("If output.hdf is not given, the tool outputs to the folder of segment_zarr.")
    exit(1)

elif len(sys.argv) == 2:
    import gt_tools
    config = gt_tools.load_config(sys.argv[1], no_db=True)
    raw_f = config["raw_file"]
    raw_zarr_ds = 'volumes/raw'
    segment_f = config["segment_file"]
    segment_ds = config["segment_ds_paintera_in"]

else:
    raw_f = sys.argv[1].split('.zarr/')[0] + '.zarr'
    raw_zarr_ds = sys.argv[1].split('.zarr/')[1]

    segment_f = sys.argv[2].split('.zarr/')[0] + '.zarr'
    segment_ds = sys.argv[2].split('.zarr/')[1]

if len(sys.argv) == 4:
    fo = sys.argv[3]
else:
    fo = segment_f.rsplit('.', 1)[0] + '.h5'

print("Creating h5 file at ", os.path.realpath(fo))
hf = h5py.File(fo, 'a')


def copy2hdf(hf, ds, daisy_ds):
    print("Copying over dataset: %s" % ds)
    hf.create_dataset(ds, data=daisy_ds.to_ndarray())
    # hf[ds].attrs['resolution'] = daisy_ds.voxel_size
    # hf[ds].attrs['offset'] = daisy_ds.roi.get_begin()

    hf[ds].attrs['resolution'] = (1, 1, 1)
    hf[ds].attrs['offset'] = daisy_ds.roi.get_begin() / daisy_ds.voxel_size


if True:
    copy2hdf(hf, 'volumes/raw', daisy.open_ds(raw_f, raw_zarr_ds))

if True:
    copy2hdf(hf, 'volumes/labels/neuron_ids', daisy.open_ds(segment_f, segment_ds))


hf.close()

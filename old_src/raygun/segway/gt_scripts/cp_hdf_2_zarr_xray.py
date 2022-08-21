# import zarr
# import h5py
import sys
import daisy

if len(sys.argv) == 1:
    print("Usage: python cp_hdf_2_zarr_xray.py file.hdf file.zarr")
    exit(1)

h5_f = sys.argv[1]
zarr_f = sys.argv[2]

# zarr_f = config["segment_file"]

src_ds = [
            'volumes/raw',
            'volumes/labels/neuron_ids',
            'volumes/labels/unlabelled',
        ]

dst_ds = [
            'volumes/raw',
            'volumes/labels/neuron_ids',
            'volumes/labels/unlabeled',
        ]

for source_dataset, dest_dataset in zip(src_ds, dst_ds):

    try:

        in_ds = daisy.open_ds(h5_f, source_dataset)

        print("in_ds.roi:",in_ds.roi)
        print("in_ds.voxel_size:",in_ds.voxel_size)
        print("in_ds.dtype:",in_ds.dtype)

        roi = in_ds.roi
        voxel_size = in_ds.voxel_size
        if voxel_size[0] == 1:
            # cannot be 1
            voxel_size = daisy.Coordinate((50, 50, 50))
            roi_offset = daisy.Coordinate(roi.get_begin()) * voxel_size
            roi_shape = daisy.Coordinate(roi.get_shape()) * voxel_size
            roi = daisy.Roi(roi_offset, roi_shape)
            print("new voxel_size:", voxel_size)
            print("new roi:", roi)

        out_ds = daisy.prepare_ds(
            zarr_f,
            dest_dataset,
            roi,
            voxel_size,
            in_ds.dtype,
            # write_roi=daisy.Roi((0, 0, 0), chunk_size),
            # write_size=chunk_size,
            compressor={'id': 'zlib', 'level': 5},
            delete=True,
            )

        out_ds[out_ds.roi] = in_ds[in_ds.roi]

    except Exception as e:
        print(e)
        continue

# h5_ds = daisy.open_ds(h5_f, 'volumes/labels/neuron_ids')

# print("Writing...")
# out_ds[out_ds.roi] = h5_ds.to_ndarray()

# # if True:
# #     ds = 'volumes/raw'
# #     print("Copying over dataset: %s"%ds)
# #     hf.create_dataset(ds,data=raw_zarr[raw_zarr_ds][:])
# #     hf[ds].attrs['resolution'] = raw_zarr[raw_zarr_ds].attrs['resolution']
# #     hf[ds].attrs['offset'] = raw_zarr[raw_zarr_ds].attrs['offset']

# # if True:
# #     ds = 'volumes/labels/neuron_ids'
# #     print("Copying over dataset: %s"%ds)
# #     hf.create_dataset(ds,data=segment_zarr[segment_ds][:])
# #     hf[ds].attrs['resolution'] = segment_zarr[segment_ds].attrs['resolution']
# #     hf[ds].attrs['offset'] = segment_zarr[segment_ds].attrs['offset']

# # hf.close()

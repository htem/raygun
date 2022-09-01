import zarr
import h5py
import sys

f = sys.argv[1]
zf = zarr.open(f, 'r')
fo = sys.argv[2]
hf = h5py.File(fo, 'a')

dsets = ['volumes/raw','volumes/segmentation_0.700']

for ds in dsets:
    print("Copying over dataset: %s"%ds)
    hf.create_dataset(ds,data=zf[ds][:])
    hf[ds].attrs['resolution'] = zf[ds].attrs['resolution']
    hf[ds].attrs['offset'] = zf[ds].attrs['offset']

hf.close()


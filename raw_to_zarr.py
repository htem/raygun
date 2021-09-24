import numpy as np
import daisy

# --- Parameters and paths --- #
volume_size = [3200, 3200, 3200]  # In voxels, xyz order
voxel_size = 120  # In nm

volume_fn = 's1GadIMSmale_180917_120nm_weakleg_.raw'


# --- Create or connect to a cloudvolume --- #
info = {'filename'=,
        'ds_name'=,
        'total_roi'=,
        'voxel_size'=,
        'dtype'=,
        'write_roi'=None,
        'write_size'=None,
        'num_channels'=1,
        'compressor'='default'
    }

print(f'Opening a cloudvolume at {cloud_path}')
vol = daisy.prepare_ds(**info)

# --- Load image volume and upload --- #
print('Loading volume into memory...')
im = np.fromfile(volume_fn, dtype=np.uint8)
print('Reshaping...')
im = im.reshape(*volume_size, order='F')

print('Uploading to google cloud...')
assert vol.shape[:3] == im.shape
vol[:, :, :] = im
print('Done.')
import gunpowder as gp
import daisy
import numpy as np
import os

import logging
logging.basicConfig(level=logging.INFO)


#CONFIG
src_path = '/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/CBxs_lobV_overview_90nm_rec5iter_db9_l20p15_.n5'
src_name = 'volumes/raw'
src_voxel_size = gp.Coordinate((90, 90, 90)) #voxel_size of source

dst_path = src_path
dst_name = 'volumes/bilinear_30nm'
dst_voxel_size = gp.Coordinate((30, 30, 30)) #voxel size to cast all data into

chunk_size = 64
compressor = {  'id': 'blosc', 
                'clevel': 3,
                'cname': 'blosclz',
                'blocksize': chunk_size
                }

num_workers = os.cpu_count()
cache_size = num_workers * 3


#MAKE PIPELINE 
# (NOTE: Is designed for isotropi cases only)
source_ds = daisy.open_ds(src_path, src_name)
total_roi = source_ds.data_roi
extents = gp.Coordinate(np.ones((len(src_voxel_size))) * chunk_size)
write_size = dst_voxel_size * extents
dtype = source_ds.dtype

daisy.prepare_ds(
    dst_path,
    dst_name,
    total_roi,
    dst_voxel_size,
    dtype,
    write_size=write_size,
    num_channels=1,
    compressor=compressor)

src_key = gp.ArrayKey('SRC')
out_key = gp.ArrayKey('OUT')

pipe = gp.ZarrSource(    # add the data source
            src_path,  # the zarr container
            {   src_key: src_name,
                },  # which dataset to associate to the array key
            {   src_key: gp.ArraySpec(interpolatable=True, voxel_size=src_voxel_size),
                }  # meta-information
        )

pipe += gp.Resample(src_key, dst_voxel_size, out_key, interp_order=1)

pipe += gp.ZarrWrite(
                dataset_names = {out_key: dst_name, },
                output_filename = dst_path,
                compression_type = compressor
                )        

scan_request = gp.BatchRequest()
# scan_request.add(src_key, write_size, src_voxel_size)
scan_request.add(out_key, write_size, dst_voxel_size)
pipe += gp.Scan(scan_request, num_workers=num_workers, cache_size=cache_size)

request = gp.BatchRequest()


#RUN
print(f'Full rendering pipeline declared for input {src_path}. Building...')
with gp.build(pipe):
    print('Starting full volume render...')
    pipe.request_batch(request)
    print('Finished.')
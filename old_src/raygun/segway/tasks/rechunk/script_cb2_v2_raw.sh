
# f="/n/groups/htem/temcagt/datasets/cb2/segmentation/zarr_volume/cb2_2.zarr"
f="/n/vast/htem/temcagt/datasets/cb2/segmentation/zarr_volume/cb2_2.zarr"
ds="raw_mipmap"
output_file='/n/vast/htem/Segmentation/cb2_v2/raw.zarr'

# python segway/tasks/rechunk/task_rechunk.py $f $ds/s8 ${output_file} $ds/s8_rechunked --num_workers 8
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s7 ${output_file} $ds/s7_rechunked --num_workers 32
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s6 ${output_file} $ds/s6_rechunked --num_workers 32
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s5 ${output_file} $ds/s5_rechunked --num_workers 32
python segway/tasks/rechunk/task_rechunk.py $f $ds/s4 ${output_file} $ds/s4_rechunked --num_workers 16
python segway/tasks/rechunk/task_rechunk.py $f $ds/s3 ${output_file} $ds/s3_rechunked --num_workers 16
python segway/tasks/rechunk/task_rechunk.py $f $ds/s2 ${output_file} $ds/s2_rechunked --num_workers 16

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# # optimized for almost isotropic voxel sizes
# # optimized for voxel size 80x64x64
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s4 $f $ds/s4_rechunked --write_size $write_size --num_workers 16

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s5 $f $ds/s5_rechunked --write_size $write_size --num_workers 16

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s6 $f $ds/s6_rechunked --write_size $write_size --num_workers 16

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s7 $f $ds/s7_rechunked --write_size $write_size --num_workers 16

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s8 $f $ds/s8_rechunked --write_size $write_size --num_workers 16


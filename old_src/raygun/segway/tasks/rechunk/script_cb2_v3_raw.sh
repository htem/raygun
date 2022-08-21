
f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
ds="volumes/raw_mipmap"

# # write_size="16 128 128"
# # # 16x128x128 = 256k voxels, optimized for neuroglancer with max_voxel_log2=18
# # # for voxel size 40x8x8
python segway/tasks/rechunk/task_rechunk.py $f $ds/s0 $f $ds/s0_rechunked --num_workers 2 --scheduling_block_size_mult 4 8 8 

f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
ds="volumes/raw_mipmap"
python segway/tasks/rechunk/task_rechunk.py $f $ds/s1 $f $ds/s1_rechunked --num_workers 2 --scheduling_block_size_mult 4 4 4 --config_hash 248296315576085943994127041216211663359


f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
ds="volumes/raw_mipmap"
python segway/tasks/rechunk/task_rechunk.py $f $ds/s2 $f $ds/s2_rechunked --num_workers 2


write_size="64 64 64"
f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
ds="volumes/raw_mipmap"
# optimized for voxel size 40x32x32
python segway/tasks/rechunk/task_rechunk.py $f $ds/s3 $f $ds/s3_rechunked --write_size $write_size --num_workers 2

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# # optimized for almost isotropic voxel sizes
# # optimized for voxel size 80x64x64
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s4 $f $ds/s4_rechunked --write_size $write_size --num_workers 2

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s5 $f $ds/s5_rechunked --write_size $write_size --num_workers 2

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s6 $f $ds/s6_rechunked --write_size $write_size --num_workers 2

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s7 $f $ds/s7_rechunked --write_size $write_size --num_workers 2

# write_size="64 64 64"
# f="/n/groups/htem/temcagt/datasets/cb2/zarr_volume/cb2_v3.zarr"
# ds="volumes/raw_mipmap"
# python segway/tasks/rechunk/task_rechunk.py $f $ds/s8 $f $ds/s8_rechunked --write_size $write_size --num_workers 2


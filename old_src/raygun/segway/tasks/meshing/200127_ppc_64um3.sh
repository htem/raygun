
f=/n/pure/htem/Segmentation/ppc_L23_100um_2.zarr
ds=volumes/super_1x2x2_segmentation_0.500
roi_offset='0 307200 1095680'
roi_shape='64000 65536 65536'
block_size='4000 8192 8192'
downsample='1 8 8'  # to 40x32x32
max_quadrics_error=1e8
min_obj_size=32  # for 40x64x64 voxels


python segway/tasks/meshing/task_meshing2.py $f $ds ${f}/meshes/segmentation_0.500/precomputed --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 0 --downsample $downsample --num_workers 16 --overwrite 0
# --block_size $block_size --roi_offset $roi_offset --roi_shape $roi_shape --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size --no_launch_workers 0 --downsample $downsample --num_workers 8
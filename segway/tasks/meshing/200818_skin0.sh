
f="/n/vast/htem/Segmentation/skin0_segmentation/200818_skin0_cutout0_setup00_140k.zarr"
ds="volumes/segmentation_0.500"
block_size='89600 179200 160000'
max_quadrics_error=1e8
min_obj_size=256  # for 16x16x16 voxels; 32 was too small

python segway/tasks/meshing/task_meshing2.py $f $ds ${f}/meshes/segmentation_0.500/precomputed --no_launch_workers 0 --num_workers 1 --overwrite 2 --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size



f=/n/groups/htem/Segmentation/xray_segmentation/outputs/xray/191211_jlc_merge_full_sc2/setup01b/300000/output.zarr
ds=volumes/segmentation_0.500_mipmap/s1
block_size='89600 179200 160000'
max_quadrics_error=1e8
min_obj_size=32  # for 40x64x64 voxels


python segway/tasks/meshing/task_meshing2.py $f $ds ${f}/meshes/segmentation_0.500/precomputed --block_size ${block_size} --no_launch_workers 0 --num_workers 1 --overwrite 2 --roi_offset 25600 76800 76800 --roi_shape 89600 179200 160000 --max_quadrics_error $max_quadrics_error --min_obj_size $min_obj_size
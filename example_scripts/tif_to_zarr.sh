# Expects tiff stack
# to convert Tiff volume into stack (sequence of Tiffs), in ImageJ: 
# Import --> Image Sequence --> Tiff Virtual Stack
# Image --> Type --> 8-bit
# Save as --> Image Sequence --> Tiff --> save in folder of name {prefix}

export PYTHONPATH=$PYTHONPATH:'/n/groups/htem/users/jlr54/raygun'
echo 'PYTHONPATH='$PYTHONPATH

# go to working directory
cd /n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX

model_prefix='train'
prefix='mCTX_17keV_30nm_512c_last256'
# aligned_dir_path="/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX/results/${model_prefix}_${prefix}"
aligned_dir_path="/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/CARE/mCTX/450p_stacks/${prefix}"
output_file="/n/groups/htem/ESRF_id16a/tomo_ML/ReducedAnglesXray/jlr54_tests/volumes/${prefix}.zarr"
output_dataset="volumes/${model_prefix}"
voxel_size='30 30 30'
roi_offset='0 0 0'
roi_shape='7680 15360 15360'
x_tile_size=512
y_tile_size=512
# section_dir_name_format="${model_prefix}_${prefix}_{:04d}.tif"
section_dir_name_format="${prefix}_{:04d}.tif"
max_voxel_count=262144 # means chunk size = 256 * 1024 (recommended for NeuroGlancer chunk loading)

python /n/groups/htem/users/jlr54/raygun/segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff_single_file.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --single_file_format 1 --max_voxel_count $max_voxel_count --no_launch_workers 0 --overwrite 0 --num_workers 4
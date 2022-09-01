# /n/groups/htem/tier2/cb3/sections/170326130159_cb3_0511/intersection/tiles_bridge9

# for i in `seq 200 511`; do
#     path=/n/groups/htem/tier2/cb3/sections/*0${i}/intersection/tiles_bridge9
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 100 199`; do
#     path=/n/groups/htem/tier2/cb3/sections/*0${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 10 99`; do
#     path=/n/groups/htem/tier2/cb3/sections/*00${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done
# for i in `seq 0 9`; do
#     path=/n/groups/htem/tier2/cb3/sections/*000${i}/intersection/tiles_with_201_511
#     if [ ! -d $path ]; then
#         echo Section $i missing!
#         continue
#     fi
#     ln -s $path $i
# done

aligned_dir_path="/n/groups/htem/temcagt/datasets/aaronsBrain/intersection/V1_aligned_tifs"
output_file='/n/groups/htem/temcagt/datasets/aaronsBrain/intersection/V1_aligned_tifs_test.n5'
output_dataset="volumes/raw"
voxel_size='45 4 4'
# write_size='16 256 256'
roi_offset='0 0 0'
# volume size
# Z: 0-249 = 250*45 = 11250
# Y: r1-53 = 53*4*2048 = 434176
# X: c1-54 = 54*4*2048 = 442368
# roi_shape='11250 434176 442368'
roi_shape='11250 32768 32768'
roi_offset='0 163840 163840'
y_tile_size=2048
x_tile_size=2048
section_dir_name_format="section_{:04d}"
missing_ok=1
bad_sections='180 202'

python segway/tasks/make_zarr_from_tiff/task_make_zarr_from_tiff.py $aligned_dir_path $y_tile_size $x_tile_size $voxel_size $output_file $output_dataset --roi_offset $roi_offset --roi_shape $roi_shape --section_dir_name_format $section_dir_name_format --no_launch_workers 1 --bad_sections $bad_sections


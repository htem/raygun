
f="/n/groups/htem/ESRF_id16a/jaspersLegCryo/bigStitcher/190312_jlc_merge_2.zarr"
ds="volumes/raw"
output_file=$f
roi_offset='2800 114688 131072'
roi_shape='44000 425984 786432'
overwrite=2

mkdir -p ${output_file}/${ds}_mipmap_new
ln -s `realpath ${f}/${ds}` ${output_file}/${ds}_mipmap_new/s0

in_ds=${ds}_mipmap_new/s0
out_ds=${ds}_mipmap_new/s1
scale_factor='1 2 2' # to 40x16x16
# python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s1
out_ds=${ds}_mipmap_new/s2
scale_factor='1 2 2' # to 40x32x32
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s2
out_ds=${ds}_mipmap_new/s3
scale_factor='2 2 2' # to 40x64x64
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s3
out_ds=${ds}_mipmap_new/s4
scale_factor='2 2 2' # to 80x128x128
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s4
out_ds=${ds}_mipmap_new/s5
scale_factor='2 2 2' # to 160x256x256
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s5
out_ds=${ds}_mipmap_new/s6
scale_factor='2 2 2' # to 320x512x512
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s6
out_ds=${ds}_mipmap_new/s7
scale_factor='2 2 2' # to 640x1024x1024
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8

in_ds=${ds}_mipmap_new/s7
out_ds=${ds}_mipmap_new/s8
scale_factor='2 2 2' # to 640x1024x1024
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --overwrite $overwrite --num_workers 8


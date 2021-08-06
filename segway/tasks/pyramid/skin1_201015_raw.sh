
f="/n/groups/htem/data/qz/200121_B2_final.n5"
ds="volumes/raw"
output_file=$f
# roi_offset='2800 114688 131072'
# roi_shape='44000 425984 786432'
# roi_shape='44000 65536 65536'

mkdir -p ${output_file}/${ds}_mipmap
ln -s `realpath ${f}/${ds}` ${output_file}/${ds}_mipmap/s0

in_ds=${ds}_mipmap/s0
out_ds=${ds}_mipmap/s1
scale_factor='2 2 2' # to 12
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor} --scheduling_block_size_mult 4 4 4

in_ds=${ds}_mipmap/s1
out_ds=${ds}_mipmap/s2
scale_factor='2 2 2' # to 24
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor} --scheduling_block_size_mult 4 4 4

in_ds=${ds}_mipmap/s2
out_ds=${ds}_mipmap/s3
scale_factor='2 2 2' # to 48
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor}

in_ds=${ds}_mipmap/s3
out_ds=${ds}_mipmap/s4
scale_factor='2 2 2' # to 96
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor}

in_ds=${ds}_mipmap/s4
out_ds=${ds}_mipmap/s5
scale_factor='2 2 2' # to 192
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor}

in_ds=${ds}_mipmap/s5
out_ds=${ds}_mipmap/s6
scale_factor='2 2 2' # to 384
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor}

in_ds=${ds}_mipmap/s6
out_ds=${ds}_mipmap/s7
scale_factor='2 2 2' # to 7..
python segway/tasks/pyramid/task_scale_pyramid2.py $output_file $in_ds $output_file $out_ds --num_workers 4 --overwrite 0 --scale_factor ${scale_factor}


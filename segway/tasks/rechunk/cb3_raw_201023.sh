
f="/n/groups/htem/temcagt/datasets/cb3/zarr/cb3.zarr"
ds="volumes/raw_mipmap"
fout="/n/f810/htem/temcagt/datasets/cb3/zarr/cb3.n5"

ds="volumes/raw_mipmap/s0"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 4 8 8 



ds="volumes/raw_mipmap/s9"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8
ds="volumes/raw_mipmap/s8"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8
ds="volumes/raw_mipmap/s7"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8
ds="volumes/raw_mipmap/s6"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 2 2 2
# 1000 blocks
ds="volumes/raw_mipmap/s5"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 4 4 4
ds="volumes/raw_mipmap/s4"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 4 4 4
ds="volumes/raw_mipmap/s3"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 4 8 8
ds="volumes/raw_mipmap/s2"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 8 8 8
ds="volumes/raw_mipmap/s1"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 8 16 16
ds="volumes/raw_mipmap/s0"
python segway/tasks/rechunk/task_rechunk.py $f $ds $fout $ds --num_workers 8 --scheduling_block_size_mult 8 16 16

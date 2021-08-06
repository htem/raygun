## Evaluating segmentation result with skeleton groundtruth

run_evaluation could plot the graph with evaluation matrices number of split_merge_error, rand, voi or just print out the coordinates of split error and merge error

### Usage

`python segway/evaluation/run_evaluation.py ${JSON_CONFIG}`

### Example usage:

`python segway/evaluation/run_evaluation.py evals/cutout4_v2_eval01.json`

### Input arguments for `run_evaluation.py`

*-p*  means number of processes to use, default set to 1  

*-i*  means build the graph with interpolation or not, default is True

*-m*  `merge`, `split`, or `both`

### Notes

A few segmentation Markers are dismissed intentionally to provide clean figure. If you don't want that or the number of threshold is more than 9, you should provide your own markers. 

Also, make sure first seg_path file in list_seg_path have complete threshold_list. For example, in first seg_path, we have segmentation_0.1 to segmentation_0.9 and rest have segmentation_0.1 to segmentation_0.7. If it is not the case, please change the j value in function *compare_threshold* in *task_05_graph_evaluation_print_error.py*

Same thing for colors, if the number of model you would like to compare is more than 10, provide your own color. 

Output path is the same as the path of config file you provided


## Print segmentation error coordinates with skeleton groundtruth

### Usage

`python segway/evaluation/print_errors.py ${CSV_PATH} ${SEGMENTATION_PATH}`

### Example usage

`python segway/evaluation/print_errors.py synapse_cutout4_skeleton.csv /n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/201904/cb2_synapse_cutout4_v2/setup12/160000/output.zarr/volumes/segmentation_0.900/`


segway_dir=/n/groups/htem/Segmentation/shared-nondev/segway2
raw_file=/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_0.n5
# raw_file=/n/groups/htem/ESRF_id16a/tomo_ML/ResolutionEnhancement/jlr54_tests/volumes/GT/CBvTopGT/CBxs_lobV_topm100um_eval_1.n5

# check if path is correct
if [[ $PYTHONPATH != */n/groups/htem/Segmentation/shared-nondev* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

scriptn=batch_`date +"%Y%m%d"`
rm $scriptn
for gt_json in $(seq -f "%02g" 0 59); do #TODO: MAKE LOOP OVER JSONS IN A FILE(?)
    command="python ${segway_dir}/tasks/segmentation/task_01.py ${gt_json}"
    command+=" Input.raw_file=${raw_file}"
    command+=" Input.experiment=ribbon${ribbon}_210103"
    echo $command >> $scriptn
done
parallel --bar -j 10 'eval CUDA_VISIBLE_DEVICES=$(({%}-1)) {}' :::: $scriptn

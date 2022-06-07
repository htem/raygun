scriptn=batch_220103
rm $scriptn
for ribbon in $(seq -f "%02g" 0 59); do
    command='python segway/tasks/task_01.py configs/test_predictions_gt_210103.config'
    command+=" Input.raw_file=gt/train_gt_211207/ribbon${ribbon}.n5"
    command+=" Input.experiment=ribbon${ribbon}_210103"
    echo $command >> $scriptn
done
parallel --bar -j 10 'eval CUDA_VISIBLE_DEVICES=$(({%}-1)) {}' :::: $scriptn
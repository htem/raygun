
export gt_json=`realpath $1`

segway_dir=/n/groups/htem/Segmentation/shared-nondev/segway

# check if path is correct
if [[ $PYTHONPATH != */n/groups/htem/Segmentation/shared-nondev* ]]; then
    echo INFO: PYTHONPATH env does not have segway... adding it
    export PYTHONPATH=$PYTHONPATH:/n/groups/htem/Segmentation/shared-nondev
fi

alias 01_copy_raw_from_zarr="python ${segway_dir}/gt_scripts/make_zarr_from_cb2_v2_zarr.py ${gt_json}"
alias 01_copy_raw_from_catmaid="python ${segway_dir}/gt_scripts/make_zarr_from_catmaid_dir.py ${gt_json}"
alias 01_copy_raw_from_catmaid_fix_bad_sections="python ${segway_dir}/gt_scripts/make_zarr_from_catmaid_dir_fix_bad_slices.py ${gt_json}"
alias 01_check_raw="python -i ${segway_dir}/gt_scripts/ng_check_raw.py ${gt_json}"

alias 07_make_zarr_gt="python ${segway_dir}/gt_scripts/make_zarr_null_gt.py ${gt_json}"
alias 07_check_zarr_gt="python -i ${segway_dir}/gt_scripts/ng_check_gt.py ${gt_json}"

alias run_skeleton_correction_no_merge="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json} --no_correct_merge"
alias run_skeleton_correction_no_split="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton.py ${gt_json} --no_correct_split"

alias test_04_run_skeleton_correction="python ${segway_dir}/gt_scripts/fix_gt_with_skeleton2.py ${gt_json}"

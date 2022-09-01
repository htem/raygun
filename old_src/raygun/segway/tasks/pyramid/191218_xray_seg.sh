
f="/n/groups/htem/Segmentation/xray_segmentation/outputs/xray/191211_jlc_merge_full_sc2/setup01b/300000/output.zarr"
output_file=$f

# for threshold in 550 575 600 625 650 675 700; do
#     ds="volumes/segmentation_0.$threshold"
#     bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file
# done


for threshold in 300 350 400 450 500 550 600 650 700 750 800 850 900; do
    ds="volumes/segmentation_0.$threshold"
    bash segway/tasks/pyramid/make_pyramid.sh $f $ds $output_file
done

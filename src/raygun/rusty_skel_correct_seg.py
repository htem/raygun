import sys
from predict import predict_task
from rusty_mws.rusty_segment_mws import get_corrected_segmentation
from funlib.geometry import Coordinate
import numpy as np
from model import neighborhood


def get_skel_correct_segmentation(
    predict_affs: bool = True,
    raw_file: str = "../../data/xpress-challenge.zarr",
    raw_dataset: str = "volumes/training_raw",
    out_file: str = "./raw_predictions.zarr",
    out_datasets=[(f"pred_affs", len(neighborhood)), (f"pred_lsds", 10)],
    iteration="latest",
    model_path="./",
) -> None:
    if predict_affs:
        # predict affs
        predict_task(
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=out_datasets,
            num_workers=4,
            # n_gpu=1,
            model_path=model_path,
        )

    # rusty mws + correction using skeletons
    get_corrected_segmentation(
        affs_file=out_file,
        affs_dataset=out_datasets[0][0],
        fragments_file=out_file,
        fragments_dataset="frag_seg",
        seeds_file=raw_file,
        seeds_dataset="volumes/training_gt_rasters",
        context=Coordinate(np.max(np.abs(neighborhood), axis=0)),
        # filter_fragments=0.57,
        filter_fragments=0,
        seeded=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        iteration = sys.argv[1]
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
    else:
        iteration = "latest"
        model_path = "./"

    get_skel_correct_segmentation(iteration=iteration, model_path=model_path)

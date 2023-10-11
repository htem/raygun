from predict import predict_task
from model import neighborhood
from funlib.geometry import Coordinate
import numpy as np
from rusty_mws.rusty_segment_mws import get_corrected_segmentation, get_pred_segmentation
import numpy as np


def get_validation_segmentation(
    iteration="latest",
    raw_file="../../data/monkey_xnh.zarr",
    raw_dataset="s46_V1_100nm_7_q3_rec_cropped",
    out_file="./validation.zarr",
) -> bool:

    affs_ds: str = f"pred_affs_{iteration}"

    predict_task( # Raw --> Affinities
        iteration=iteration,
        raw_file=raw_file,
        raw_dataset=raw_dataset,
        out_file=out_file,
        out_datasets=[(affs_ds, 12)],
        num_workers=20,
        n_gpu=3)


    context = Coordinate(np.max(np.abs(neighborhood), axis=0))

    get_pred_segmentation(affs_file=out_file,
                          affs_dataset=affs_ds,
                          fragments_file=out_file,
                          fragments_dataset="frag_seg",
                          context=context,
                          filter_fragments=0.4,
                          adj_bias=2.,
                          lr_bias=-.5)
    
    return True


if __name__ == "__main__":
    get_validation_segmentation()
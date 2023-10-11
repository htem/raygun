import gunpowder as gp
import logging
import math
import numpy as np
import torch
from lsd.train.gp import AddLocalShapeDescriptor

from tqdm import trange

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

from ..models import MTLSDModel
from ...rusty_skel_correct_seg import get_skel_correct_segmentation

raw_file = "../../data/monkey_xnh.zarr"
raw_dataset = "s46_V1_100nm_7_q3_rec_cropped"
out_file = "./raw_predictions.zarr"
iteration = "latest"


def pipeline(iterations, warmup=5000, save_every=1000):
    
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK") 
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")

    model = MTLSDModel(unet, num_fmaps)
    loss = WeightedMTLSD_MSELoss()#aff_lambda=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))

    increase = 8 * 3
    input_shape = [132 + increase] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]
    print(input_shape, output_shape)

    voxel_size = gp.Coordinate((33,) * 3)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    # Zarr sources
    predicted_source = (
        gp.ZarrSource(
            raw_file,
            {
                raw: raw_dataset,
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
            },
        ),
        gp.ZarrSource(
            out_file,
            {
                labels: f"pred_seg",
                labels_mask: f"pred_labels_mask",
                unlabelled: f"pred_unlabelled_mask",
            },
            {
                labels: gp.ArraySpec(interpolatable=False),
                labels_mask: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            },
        ),
    ) + gp.MergeProvider()
    predicted_source += gp.MergeProvider()

    predicted_source += gp.RandomLocation(mask=labels_mask, min_masked=0.5)

    gt_source = gp.ZarrSource(
        raw_file,
        {
            raw: raw_dataset,
            labels: f"volumes/training_gt_labels",
            labels_mask: f"volumes/training_labels_mask",
            unlabelled: f"volumes/training_unlabelled_mask",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
        },
    )
    gt_source += gp.MergeProvider()
    gt_source += gp.RandomLocation(mask=labels_mask, min_masked=0.5)
   
    def get_training_pipeline():
        
        request = gp.BatchRequest()

        request.add(raw, input_size)
        request.add(labels, output_size)
        request.add(labels_mask, output_size)
        request.add(gt_affs, output_size)
        request.add(gt_lsds, output_size)
        request.add(affs_weights, output_size)
        request.add(gt_affs_mask, output_size)
        request.add(gt_lsds_mask, output_size)
        request.add(unlabelled, output_size)
        request.add(pred_affs, output_size)
        request.add(pred_lsds, output_size)

        training_pipeline = gp.Normalize(raw)
        training_pipeline += gp.Pad(raw, None)
        training_pipeline += gp.Pad(labels, context)
        training_pipeline += gp.Pad(labels_mask, context)
        training_pipeline += gp.Pad(unlabelled, context)

        training_pipeline += gp.ElasticAugment(
            control_point_spacing=[30, 30, 30],
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            subsample=8,
        )

        training_pipeline += gp.SimpleAugment()

        training_pipeline += RandomNoiseAugment(raw)

        training_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

        training_pipeline += SmoothArray(raw, (0.0,1.0))

        training_pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            sigma=10 * 33,
            lsds_mask=gt_lsds_mask,
            unlabelled=unlabelled,
            downsample=2,
        )

        training_pipeline += gp.GrowBoundary(labels, mask=unlabelled)

        training_pipeline += gp.AddAffinities(
            affinity_neighborhood=neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabelled,
            affinities_mask=gt_affs_mask,
            dtype=np.float32
        )

        training_pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

        training_pipeline += gp.Unsqueeze([raw])
        training_pipeline += gp.Stack(1)

        training_pipeline += gp.PreCache(cache_size=40, num_workers=10)

        training_pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: gt_lsds_mask,
                3: pred_affs,
                4: gt_affs,
                5: affs_weights,
            },
            outputs={0: pred_lsds, 1: pred_affs},
            save_every=save_every,
            log_dir="log",
        )

        training_pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt_lsds: "gt_lsds",
                unlabelled: "unlabelled",
                pred_lsds: "pred_lsds",
                gt_affs: "gt_affs",
                pred_affs: "pred_affs",
                affs_weights: "affs_weights"
            },
            dataset_dtypes={
                gt_affs: np.float32
            },
            output_filename="batch_{iteration}.zarr",
            every=save_every,
        )

        return training_pipeline, request

    # First iterations are warmup on voxel data
    if (
        warmup is None
    ):  # Allows to do initial segmentation with existing model checkpoints
        # Make segmentation predictions
        get_skel_correct_segmentation(predict_affs=True)
        model.train()
    elif warmup > 0:
        training_pipeline, request = get_training_pipeline()
        pipeline = (
            gt_source
            + training_pipeline
        )

        with gp.build(pipeline):
            for i in trange(warmup):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_skel_correct_segmentation()
        model.train()

    # Add segmentation predictions to training pipeline
    # Then repeat, scaling up the prediction usage
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"Rinse & Repeat @ ratio: {ratio}")
        training_pipeline, request = get_training_pipeline()
        pipeline = (gt_source, predicted_source) + gp.RandomProvider(
            probabilities=[1 - ratio, ratio]
        )
        pipeline += training_pipeline
        with gp.build(pipeline):
            for i in trange(iterations):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_skel_correct_segmentation()
        model.train()


if __name__ == "__main__":
    pipeline(iterations=10, warmup=20000, save_every=10000)

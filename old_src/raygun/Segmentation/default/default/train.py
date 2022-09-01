from __future__ import print_function
import sys
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder-1.2.2-220114')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder.210911')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/gunpowder-1.3')
from gunpowder.jax import Train as JaxTrain

from gunpowder import *
import daisy
from reject import Reject
import os
import math
import json
import numpy as np
import logging
from mknet import create_network

logging.basicConfig(level=logging.INFO)

raw = ArrayKey('RAW')
labels = ArrayKey('GT_LABELS')
labels_mask = ArrayKey('GT_LABELS_MASK')
unlabeled_mask = ArrayKey('GT_UNLABELED_MASK')
# relevant_mask = ArrayKey('GT_RELEVANT_MASK')
affs = ArrayKey('PREDICTED_AFFS')
gt_affs = ArrayKey('GT_AFFINITIES')
gt_affs_mask = ArrayKey('GT_AFFINITIES_MASK')
gt_affs_scale = ArrayKey('GT_AFFINITIES_SCALE')
affs_gradient = ArrayKey('AFFS_GRADIENT')

global cache_size
global snapshot_every

neighborhood = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-3, 0, 0],
    [0, -3, 0],
    [0, 0, -3],
    [-9, 0, 0],
    [0, -9, 0],
    [0, 0, -9],
    [-27, 0, 0],
    [0, -27, 0],
    [0, 0, -27]
])


def train_affinity(dense_samples,
                raw_ds,
                labels_ds,
                labels_mask_ds,
                unlabeled_mask_ds,
                max_iteration,
                num_workers,
                batch_size,
                model=None,                
                cache_size=40,
                snapshot_every=1000,
                save_every=10000,
                keep_n_checkpoints=10
                ):
    if model is None:
        model = create_network()
    
    for folder in ['log', 'metrics']: # 'checkpoints'
        if not os.path.exists(folder):
            os.makedirs(folder)

    print(f'cache_size={cache_size}')

    trained_until = 0
    if trained_until >= max_iteration:
        return

    with open('train_net.json', 'r') as f:
        config = json.load(f)

    voxel_size = daisy.open_ds(dense_samples[0], raw_ds).voxel_size #Coordinate((30, 30, 30))
    input_size = Coordinate(config['input_shape'])*voxel_size
    output_size = Coordinate(config['output_shape'])*voxel_size

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(unlabeled_mask, output_size)
    # request.add(relevant_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({
        affs: request[labels],
        affs_gradient: request[labels],
        #gt_affs: request[labels],
    })

    padding_voxels = 32
    source_padding = Coordinate([padding_voxels*voxel for voxel in voxel_size])

    def create_source(sample):

        src = ZarrSource(
                sample,
                datasets={
                    raw: raw_ds,
                    # gt_affs: 'volumes/affs_mipmap2',
                    labels: labels_ds,
                    labels_mask: labels_mask_ds,
                    unlabeled_mask: unlabeled_mask_ds,
                    # relevant_mask: 'volumes/labels/relevant_mask' if sample not in background_samples else 'volumes/labels/unlabeled',
                },
                array_specs={
                    raw: ArraySpec(interpolatable=True),
                    # gt_affs: ArraySpec(interpolatable=True),
                    labels: ArraySpec(interpolatable=False),
                    labels_mask: ArraySpec(interpolatable=False),
                    unlabeled_mask: ArraySpec(interpolatable=False),
                    # relevant_mask: ArraySpec(interpolatable=False),
                }
            )

        src += Pad(raw, None)
        src += Pad(labels, source_padding)
        src += Pad(labels_mask, source_padding)
        src += Pad(unlabeled_mask, source_padding)
        # src += Pad(relevant_mask, source_padding)

        src += RandomLocation(
            # mask=unlabeled_mask,
            # min_masked=0.1,
            )

        src += Reject(
            mask=unlabeled_mask,
            min_masked=0.1,
        )

        src += Normalize(raw)

        return src

    data_sources = tuple(
        create_source(sample) for sample in dense_samples
    )

    train_pipeline = (
        data_sources +
        # RandomProvider(probabilities=total_sample_weights) +
        RandomProvider() +
        ElasticAugment(
            control_point_spacing=[10, 10, 10],  # worked for 30nm
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi/2.0],
            subsample=2,
            ) +

        SimpleAugment() +

        IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1) +
        NoiseAugment(raw, var=0.01) +
        GrowBoundary(labels, labels_mask, steps=1) +
        AddAffinities(
            neighborhood,
            labels=labels,
            affinities=gt_affs,
            labels_mask=labels_mask,
            unlabelled=unlabeled_mask,
            affinities_mask=gt_affs_mask) +

        # doesn't work for fractional GT affs
        BalanceLabels(
            gt_affs,
            gt_affs_scale,
            gt_affs_mask,
            ) +

        # DefectAugment(raw, prob_missing=0.005, max_consecutive_missing=3) +
        IntensityScaleShift(raw, 2, -1) +
        # add raw "channel"
        Unsqueeze([raw]) +
        # add batch
        Stack(batch_size) +
        # Unsqueeze([raw, gt_affs, gt_affs_scale]) +
        PreCache(
            cache_size=cache_size,
            num_workers=num_workers-1) +
        JaxTrain(
            model=model,
            inputs={
                'raw': raw,
                'gt': gt_affs,
                'mask': gt_affs_scale,
            },
            outputs={
                'affs': affs,
                'grad': affs_gradient
            },
            log_dir=f'log/{raw_ds.split("/")[-1]}',
            checkpoint_basename=f'{raw_ds.split("/")[-1]}',
            save_every=save_every,
            keep_n_checkpoints=keep_n_checkpoints,
            ) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        Snapshot({
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                gt_affs: 'volumes/gt_affinities',
                affs: 'volumes/pred_affinities',
                labels_mask: 'volumes/labels/mask',
                unlabeled_mask: 'volumes/labels/unlabeled',
                affs_gradient: 'volumes/affs_gradient'
            },
            # dataset_dtypes={
            #     labels: np.uint64
            # },
            every=snapshot_every,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) #+
        # PrintProfilingStats(every=100)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as config_file:
        kwargs = json.load(config_file)
    train_affinity(**kwargs)

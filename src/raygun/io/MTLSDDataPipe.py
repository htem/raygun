import math
import daisy
import gunpowder as gp
import numpy as np

from raygun.io import BaseDataPipe
from raygun.utils import passing_locals, calc_max_padding

from lsd.train.gp import AddLocalShapeDescriptor


class MTLSDDataPipe(BaseDataPipe):
    def __init__(
        self,
        srcs,
        ndims,
        output_size,
        voxel_size,  # TODO: Add capacity to train on datasets from multiple voxel-sizes, resampling as necessary
        neighborhood,
        lsd_kwargs=None,
        batch_size=1,
        pad=False,
        random_location_kwargs=None,
        reject_kwargs=None,
        **kwargs,
    ):
        super().__init__(**passing_locals(locals()))

        # self.src_voxel_size = daisy.open_ds(
        #     self.src["path"], self.src["real_name"]
        # ).voxel_size

        # declare arrays to use in the pipelines
        array_names = [
            "raw",
            "labels",
            "labels_mask",
            "cell_mask",
            "gt_affs",
            "pred_affs",
            "affs_mask",
            "affs_weights",
        ]

        if self.lsd_kwargs is not None:
            array_names += ["gt_lsds", "pred_lsds", "lsds_weights"]

        self.arrays = {}
        for array in array_names:
            array_key = gp.ArrayKey(array.upper())
            setattr(self, array, array_key)  # add ArrayKeys to object
            self.arrays[array] = array_key

        # setup data sources
        self.get_sources()

        # add augmentations
        self.augment_axes = list(np.arange(3)[-ndims:])
        self.augment = None

        if (
            hasattr(self, "elastic_aug1_kwargs")
            and self.elastic_aug1_kwargs is not None
        ):
            self.augment = gp.ElasticAugment(
                spatial_dims=ndims,
                rotation_interval=(0, math.pi / 2),
                **self.elastic_aug1_kwargs,
            )

        if self.augment is None:
            self.augment = gp.SimpleAugment(
                mirror_only=self.augment_axes, transpose_only=self.augment_axes
            )
        else:
            self.augment += gp.SimpleAugment(
                mirror_only=self.augment_axes, transpose_only=self.augment_axes
            )

        if (
            hasattr(self, "elastic_aug2_kwargs")
            and self.elastic_aug2_kwargs is not None
        ):
            self.augment += gp.ElasticAugment(
                spatial_dims=ndims,
                rotation_interval=(0, math.pi / 2),
                **self.elastic_aug2_kwargs,
            )

        if (
            hasattr(self, "intensity_aug_kwargs")
            and self.intensity_aug_kwargs is not None
        ):
            self.augment += gp.IntensityAugment(
                self.arrays["raw"],
                **self.intensity_aug_kwargs,
            )

        if self.lsd_kwargs is not None:
            self.preprocess = AddLocalShapeDescriptor(
                self.arrays["labels"],
                self.arrays["gt_lsds"],
                lsds_mask=self.arrays["lsds_weights"],
                **self.lsd_kwargs,
            )

            self.preprocess += gp.AddAffinities(
                affinity_neighborhood=neighborhood,
                labels=self.arrays["labels"],
                affinities=self.arrays["gt_affs"],
                labels_mask=self.arrays["labels_mask"],
                affinities_mask=self.arrays["affs_mask"],
                dtype=np.float32,
            )

        else:

            self.preprocess = gp.AddAffinities(
                affinity_neighborhood=neighborhood,
                labels=self.arrays["labels"],
                affinities=self.arrays["gt_affs"],
                labels_mask=self.arrays["labels_mask"],
                affinities_mask=self.arrays["affs_mask"],
                dtype=np.float32,
            )

        self.preprocess += gp.BalanceLabels(
            self.arrays["gt_affs"],
            self.arrays["affs_weights"],
            self.arrays["affs_mask"],
        )

        self.preprocess += gp.IntensityScaleShift(self.arrays["raw"], 2, -1)

        # add "channel" dimensions if neccessary, else use z dimension as channel
        if ndims == len(self.voxel_size):
            self.unsqueeze = gp.Unsqueeze([self.arrays["raw"]])
        else:
            self.unsqueeze = None

    def get_sources(self, names=["raw", "labels", "labels_mask", "cell_mask"]):
        if not isinstance(self.srcs, list):
            self.srcs = [self.srcs]

        src_specs = {
            self.raw: gp.ArraySpec(interpolatable=True),
            self.labels: gp.ArraySpec(interpolatable=False),
            self.labels_mask: gp.ArraySpec(interpolatable=False),
            self.cell_mask: gp.ArraySpec(interpolatable=False),
        }

        self.sources = []
        for src in self.srcs:
            src_names = {getattr(self, k): v for k, v in src.items() if k in names}
            source = self.get_source(
                path=src["path"],
                src_names=src_names,
                src_specs={k: src_specs[k] for k in src_names.keys()},
            )
            self.sources.append(source)

        self.sources = tuple(self.sources)
        if len(self.sources) > 1:
            self.source = self.sources + gp.MergeProvider()
        else:
            self.source = self.sources[0]

        self.source += gp.Normalize(self.raw)

        if self.pad:
            labels_padding = calc_max_padding(
                self.output_size, self.voxel_size, self.neighborhood, self.sigma
            )
            self.source += gp.Pad(self.raw, None)
            self.source += gp.Pad(self.labels, labels_padding)
            self.source += gp.Pad(self.labels_mask, labels_padding)

        if self.random_location_kwargs is not None:
            self.source += gp.RandomLocation(
                mask=self.cell_mask, **self.random_location_kwargs
            )
        else:
            self.source += gp.RandomLocation()

        # use a mask to ensure batches see enough cells
        if self.reject_kwargs is not None:
            self.source += gp.Reject(mask=self.cell_mask, **self.reject_kwargs)

        if hasattr(self, "grow_boundary") and self.grow_boundary:
            self.source += gp.GrowBoundary(self.labels)

    def prenet_pipe(self, mode: str = "train"):
        # Make pre-net datapipe
        prenet_pipe = self.source
        sections = [
            "augment",
            "preprocess",
            "unsqueeze",
            gp.Stack(self.batch_size),
        ]

        for section in sections:
            if (
                isinstance(section, str)
                and hasattr(self, section)
                and getattr(self, section) is not None
            ):
                prenet_pipe += getattr(self, section)
            elif isinstance(section, gp.nodes.BatchFilter):
                prenet_pipe += section

        return prenet_pipe

    def postnet_pipe(self, batch_size=None):
        # Make post-net data pipes
        if batch_size is None:
            batch_size = self.batch_size
        # remove "channel" dimensions if neccessary
        postnet_pipe = gp.IntensityScaleShift(self.arrays["raw"], 0.5, 0.5)

        if self.ndims == len(self.voxel_size):
            postnet_pipe += gp.Squeeze(
                [
                    self.arrays["raw"],
                ],
                axis=1,
            )  # remove channel dimension for grayscale

        if batch_size == 1:
            postnet_pipe += gp.Squeeze(
                [
                    self.arrays["raw"],
                    self.arrays["gt_lsds"],
                    self.arrays["gt_affs"],
                    self.arrays["pred_lsds"],
                    self.arrays["pred_affs"],
                ],
                axis=0,
            )  # remove batch dimension if necessary

        return postnet_pipe

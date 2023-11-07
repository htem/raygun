import itertools
import logging
from matplotlib import pyplot as plt
import torch
import daisy

import gunpowder as gp
import numpy as np

torch.backends.cudnn.benchmark = True

from raygun.io import ACLSDDataPipe
from raygun.torch.models import ACLSDModel
from raygun.torch.losses import WeightedMSELoss_ACLSD
from raygun.torch.optimizers import get_base_optimizer
from raygun.torch.systems import BaseSystem


class ACLSD(BaseSystem):
    def __init__(self, config=None):
        super().__init__(
            default_config="../default_configs/default_ACLSD_conf.json",
            config=config,
        )
        self.logger = logging.Logger(__name__, "INFO")

        if self.ndims is None:
            self.ndims = sum(np.array(self.voxel_size) == np.min(self.voxel_size))

        self.neighborhood = np.array(self.neighborhood)
        if self.n_diagonals > 0:
            pos_diag = np.round(
                self.n_diagonals
                * np.sin(np.linspace(0, np.pi, num=self.n_diagonals, endpoint=False))
            )
            neg_diag = np.round(
                self.n_diagonals
                * np.cos(np.linspace(0, np.pi, num=self.n_diagonals, endpoint=False))
            )
            stacked_diag = np.stack([0 * pos_diag, pos_diag, neg_diag], axis=-1)
            self.neighborhood = np.concatenate(
                [self.neighborhood, stacked_diag]
            ).astype(np.int8)

        self.voxel_size = gp.Coordinate(self.voxel_size)
        self.output_size = gp.Coordinate(self.output_shape) * gp.Coordinate(
            self.voxel_size
        )
        self.input_size = gp.Coordinate(self.input_shape) * gp.Coordinate(
            self.voxel_size
        )

    def setup_networks(self):
        pass

    def setup_model(self):
        self.model = ACLSDModel(**self.model_kwargs)

    def setup_optimization(self):
        self.optimizer = get_base_optimizer(self.optim_type)(
            params=self.model.parameters(), **self.optim_kwargs
        )

        self.loss = WeightedMSELoss_ACLSD(**self.loss_kwargs)

    def setup_datapipes(self):
        kws = [
            "ndims",
            "output_size",
            "voxel_size",
            "neighborhood",
            "batch_size",
            "pad",
            "random_location_kwargs",
            "reject_kwargs",
            "grow_boundary",
        ]

        dp_kwargs = {"srcs": self.sources}
        for kw in kws:
            if hasattr(self, kw) and getattr(self, kw) is not None:
                dp_kwargs[kw] = getattr(self, kw)

        self.datapipes = {"main": ACLSDDataPipe(**dp_kwargs)}

        self.arrays = self.datapipes["main"].arrays

    def make_request(self, mode: str = "train"):
        # create request
        request = gp.BatchRequest()
        for array_name, array in self.arrays.items():
            if array_name == "raw":
                extents = self.input_size
            else:
                extents = self.output_size

            request.add(array, extents, self.voxel_size)

        return request


if __name__ == "__main__":
    system = ACLSD(config="./train_conf.json")
    system.logger.info("MTLSD system loaded. Training...")
    _ = system.train()
    system.logger.info("Done training!")

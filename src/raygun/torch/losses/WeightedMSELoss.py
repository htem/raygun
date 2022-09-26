from raygun.torch.losses import BaseCompetentLoss
import torch


class WeightedMSELoss(BaseCompetentLoss):
    def __init__(self):
        super().__init__()

    def _calc_loss(self, pred, target, weights):

        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:

            mask = torch.masked_select(scaled, torch.gt(weights, 0))

            loss = torch.mean(mask)

        else:

            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        prediction,
        target,
        weights,
    ):

        loss = self._calc_loss(prediction, target, weights)

        return loss

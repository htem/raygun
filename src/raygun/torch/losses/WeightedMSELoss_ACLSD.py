from raygun.torch.losses.BaseCompetentLoss import BaseCompetentLoss
from raygun.utils import passing_locals
import torch


class WeightedMSELoss_MTLSD(BaseCompetentLoss):
    def __init__(self, **kwargs):
        super().__init__(**passing_locals(locals()))
        self.data_dict = {}

    def _calc_loss(self, pred, target, weights=None):
        if weights is not None:
            scaled = weights * (pred - target) ** 2
        else:
            scaled = (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:

            mask = torch.masked_select(scaled, torch.gt(weights, 0))

            loss = torch.mean(mask)

        else:

            loss = torch.mean(scaled)

        return loss

    def add_log(self, writer, step):
        # add loss values
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)

        # add loss input image examples
        for name, data in self.data_dict.items():
            if len(data.shape) > 3:  # pull out batch dimension if necessary
                img = data[0].squeeze()
            else:
                img = data.squeeze()

            is_rgb = False
            if len(img.shape) > 3:  # check for rgb
                mid = img.shape[1] // 2  # for 3D volume
                img = img[:3, mid, ...]
                is_rgb = True
            elif img.shape[0] != img.shape[1]:
                img = img[:3, ...]
                is_rgb = True
            elif len(img.shape) == 3:
                mid = img.shape[0] // 2  # for 3D volume
                img = img[mid]

            if (
                (img.min() < 0) and (img.min() >= -1.0) and (img.max() <= 1.0)
            ):  # scale img to [0,1] if necessary
                img = (img * 0.5) + 0.5
            if is_rgb:
                writer.add_image(name, img, global_step=step, dataformats="CHW")
            else:
                writer.add_image(name, img, global_step=step, dataformats="HW")

    def forward(
        self,
        pred_lsds=None,
        gt_lsds=None,
        lsds_weights=None,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
        pred_affs_ac=None,
    ):
        self.data_dict.update(
            {
                "pred_lsds": pred_lsds.detach(),
                "gt_lsds": gt_lsds.detach(),
                "lsds_weights": lsds_weights.detach(),
                "pred_affs": pred_affs.detach(),
                "gt_affs": gt_affs.detach(),
                "affs_weights": affs_weights.detach(),
                "pred_affs_ac": pred_affs_ac.detach(),
            }
        )
        try:
            lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
            aff_loss = self._calc_loss(pred_affs, gt_affs, affs_weights)
        except:
            lsd_loss = aff_loss = 0.

        try:
            ac_aff_loss = self._calc_loss(pred_affs_ac, gt_affs)
        except:
            ac_aff_loss = 0.

        self.loss_dict = {"LSDs": lsd_loss.detach(), "Affinities1": aff_loss.detach(), "Affinities2": ac_aff_loss}

        return lsd_loss + aff_loss + ac_aff_loss

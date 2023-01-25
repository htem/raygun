import torch


class CARELoss(torch.nn.MSELoss):
    """
    Loss module for a CARE model/system
    """
    def __init__(self) -> None:
        super().__init__()
        self.data_dict = {}

    def _calc_loss(self, pred, target, weights) -> torch.Tensor:
        """
        
        Internal function to help calculate basic MSE loss

        Returns:
            torch.Tensor: calculated loss given the MSE of pred-targ or the averge of mask value
        """

        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:

            mask: torch.Tensor = torch.masked_select(scaled, torch.gt(weights, 0))

            loss = torch.mean(mask)

        else:

            loss: torch.Tensor = torch.mean(scaled)

        return loss

    def add_log(self, writer, step) -> None:
        """
        Adds a logger to track generated image progress throughout the training process

        Returns:
            None
        """
        for key, loss in self.loss_dict.items():
            writer.add_scalar(key, loss, step)

        # add loss input image examples
        for name, data in self.data_dict.items():
            if len(data.shape) > 3:  # pull out batch dimension if necessary
                img = data[0].squeeze()
            else:
                img = data.squeeze()

            is_rgb: bool = False
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
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
    ) -> torch.Tensor:
        """
        Forward loss pass in order to calculate MSE on affs tensor

        Returns:
            torch.Tensor: calculated loss tensor 
        """
        self.data_dict.update(
            {
                "pred_affs": pred_affs.detach(),
                "gt_affs": gt_affs.detach(),
                "affs_weights": affs_weights.detach(),
            }
        )

        aff_loss: torch.Tensor = self._calc_loss(pred_affs, gt_affs, affs_weights)

        self.loss_dict: dict[str, torch.Tensor] = {"Affinities": aff_loss.detach()}

        return aff_loss

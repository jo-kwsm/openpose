import torch.nn as nn
import torch.nn.functional as F


class OpenPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        saved_for_loss,
        heatmap_target,
        heat_mask,
        paf_target,
        paf_mask
    ):
        loss = 0

        for i in range(6):
            pred1 = saved_for_loss[2 * i] *paf_mask
            gt1 = paf_target.float() * paf_mask

            pred2 = saved_for_loss[2 * i + 1] * heat_mask
            gt2 = heatmap_target.float() * heat_mask

            loss += F.mse_loss(pred1, gt1, reduction="mean") + \
                F.mse_loss(pred2, gt2, reduction="mean")

        return loss

from __future__ import annotations

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceLossBinary(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def make_smp_unet(encoder_name: str) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


def batch_iou_dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    dice_den = preds.sum(dim=dims) + targets.sum(dim=dims)

    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2 * intersection + eps) / (dice_den + eps)).mean().item()
    return iou, dice


def binary_scores(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred = (pred_mask > 0.5).astype(np.uint8)
    true = (true_mask > 0.5).astype(np.uint8)

    intersection = float((pred & true).sum())
    union = float((pred | true).sum())
    pred_sum = float(pred.sum())
    true_sum = float(true.sum())

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + true_sum + eps)
    return iou, dice


BCE_LOSS = nn.BCEWithLogitsLoss()
DICE_LOSS = DiceLossBinary()

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_weighted_bce_from_class_weights(class_weights: Dict[int, float], device: torch.device) -> nn.Module:
    pos_weight_value = class_weights[1] / max(class_weights[0], 1e-8)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")


def binary_entropy_regularization(gate_weights: torch.Tensor, paired_mask: torch.Tensor) -> torch.Tensor:
    if gate_weights.numel() == 0 or paired_mask.sum() == 0:
        return torch.zeros((), device=gate_weights.device)
    entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-8))).sum(dim=1)
    return (entropy * paired_mask).sum() / paired_mask.sum().clamp_min(1.0)


def heteroscedastic_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    base_loss = criterion(logits, targets)
    precision = torch.exp(-uncertainty).clamp(max=50.0)
    weighted = precision * base_loss + uncertainty
    return weighted.mean()


def standard_bce_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    base_loss = criterion(logits, targets)
    if base_loss.ndim == 0:
        return base_loss
    return base_loss.mean()


def multimodal_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    gate_weights: torch.Tensor,
    paired_mask: torch.Tensor,
    criterion: nn.Module,
    gate_entropy_weight: float = 0.0,
    uncertainty_loss_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    cls_loss = standard_bce_loss(logits, targets, criterion)
    uncertainty_loss = heteroscedastic_bce_loss(logits, targets, uncertainty, criterion)
    gate_reg = binary_entropy_regularization(gate_weights, paired_mask)
    total = cls_loss
    if uncertainty_loss_weight > 0:
        total = total + uncertainty_loss_weight * uncertainty_loss
    if gate_entropy_weight > 0:
        total = total - gate_entropy_weight * gate_reg
    return {
        "total_loss": total,
        "cls_loss": cls_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
        "gate_reg": gate_reg.detach(),
    }

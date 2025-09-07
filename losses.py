import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'compute_prototype_loss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss

def compute_prototype_loss(features, targets, prototypes, num_prototypes=1):
    """
    Compute prototype loss (cosine similarity) between features and class prototypes.

    Args:
        features: Tensor of shape (B, C) or (B, C, H, W)
        targets: Tensor of shape (B, H, W) or (B,) with integer class labels
        prototypes: Tensor of shape (num_classes * num_prototypes, C) or (num_classes * num_prototypes, C, 1)
        num_prototypes: Number of prototypes per class

    Returns:
        prototype_loss: scalar tensor
    """
    device = features.device

    # Flatten spatial dimensions if present
    if features.dim() > 2:
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    if targets.dim() > 1:
        targets = targets.view(-1)  # (B*H*W,)

    # For binary segmentation, we have only one class of interest
    num_classes = 1

    # Normalize features and prototypes for cosine similarity
    features = F.normalize(features, dim=1, eps=1e-8)
    prototypes = F.normalize(prototypes.view(prototypes.shape[0], -1), dim=1, eps=1e-8)  # ensure 2D

    prototype_loss = 0.0
    active_classes = 0

    for cls in range(num_classes):
        start = cls * num_prototypes
        end = (cls + 1) * num_prototypes
        cls_protos = prototypes[start:end]  # (num_prototypes, C)

        # Debug prints
        # print(f"[DEBUG] targets.shape: {targets.shape}")
        # print(f"[DEBUG] targets unique values: {targets.unique()}")
        
        mask = (targets == cls)
        if mask.sum() == 0:
            continue  # no samples for this class in batch

        active_classes += 1
        cls_feats = features[mask]  # (N, C)

        # cosine similarity between features and prototypes
        sim = torch.mm(cls_feats, cls_protos.t())  # (N, num_prototypes)
        max_sim, _ = sim.max(dim=1)  # max over prototypes

        cls_loss = (1 - max_sim).mean()
        prototype_loss += cls_loss

    if active_classes == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return prototype_loss / active_classes

    # return prototype_loss / num_classes
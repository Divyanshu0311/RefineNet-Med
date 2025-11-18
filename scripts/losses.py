# losses.py (updated)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

# ---------------------- #
#  Basic Loss Functions  #
# ---------------------- #

def dice_loss(pred, target, smooth=1.):
    """
    pred: logits (B,1,H,W)
    target: binary mask (B,1,H,W)
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    denom = pred_flat.sum(1) + target_flat.sum(1)
    loss = 1 - ((2. * intersection + smooth) / (denom + smooth))
    return loss.mean()

bce = nn.BCEWithLogitsLoss()

def bce_dice_loss(pred, target, dice_w=1.0, bce_w=1.0):
    return bce(pred, target) * bce_w + dice_loss(pred, target) * dice_w

def boundary_loss(pred_logits, target):
    """
    Compute boundary-aware loss via distance transform.
    Returns a tensor located on the same device as pred_logits.
    """
    device = pred_logits.device
    pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch = pred_prob.shape[0]
    loss = 0.0
    for i in range(batch):
        p = pred_prob[i, 0]
        t = target_np[i, 0]
        t_edge = ndimage.distance_transform_edt(1 - t)
        loss += (p * t_edge).mean()
    loss_tensor = torch.tensor(loss / batch, dtype=torch.float32, device=device)
    return loss_tensor

def generator_adversarial_loss(pred_fake):
    # L2 loss toward 1
    return torch.mean((pred_fake - 1.)**2)

def discriminator_adversarial_loss(pred_real, pred_fake):
    # L2 GAN losses
    return torch.mean((pred_real - 1.)**2) + torch.mean(pred_fake**2)


# ----------------------------- #
#  PINN-Inspired Regularization #
# ----------------------------- #
def pinn_regularization_loss(pred, img, lambda_grad=0.5, lambda_edge=0.5):
    """
    Physics-inspired regularization:
      - Gradient smoothness: penalizes abrupt mask changes
      - Edge alignment: encourages mask edges to follow image edges
    """
    pred_prob = torch.sigmoid(pred)

    # --- Gradient Smoothness (like ∥∇mask∥¹) ---
    dx = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])
    dy = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
    grad_smoothness = (dx.mean() + dy.mean())

    # --- Edge Alignment (|∇img - ∇mask|) ---
    gray = torch.mean(img, dim=1, keepdim=True)  # convert to grayscale
    gx_img = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    gy_img = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    gx_pred = torch.abs(pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1])
    gy_pred = torch.abs(pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :])
    edge_alignment = (torch.abs(gx_img - gx_pred).mean() + torch.abs(gy_img - gy_pred).mean())

    return lambda_grad * grad_smoothness + lambda_edge * edge_alignment


# ----------------------------- #
#  Full Combined Hybrid + PINN  #
# ----------------------------- #
def hybrid_pinn_loss(pred, target, img, dice_w=1.0, bce_w=1.0, boundary_w=0.5, pinn_w=0.3):
    """
    Combined loss with PINN-inspired regularization.
    pred: logits from network
    target: ground-truth mask (0/1)
    img: original image (B,3,H,W)
    """
    loss_seg = bce_dice_loss(pred, target, dice_w, bce_w)
    loss_boundary = boundary_loss(pred, target)
    loss_pinn = pinn_regularization_loss(pred, img)

    total = loss_seg + boundary_w * loss_boundary + pinn_w * loss_pinn
    return total

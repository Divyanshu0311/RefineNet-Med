import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

def dice_loss(pred, target, smooth=1.):
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
    pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch = pred_prob.shape[0]
    loss = 0.0
    for i in range(batch):
        p = pred_prob[i,0]
        t = target_np[i,0]
        t_edge = ndimage.distance_transform_edt(1 - t)
        loss += (p * t_edge).mean()
    return torch.tensor(loss / batch, dtype=torch.float32)

def generator_adversarial_loss(pred_fake):
    return torch.mean((pred_fake - 1.)**2)

def discriminator_adversarial_loss(pred_real, pred_fake):
    return torch.mean((pred_real - 1.)**2) + torch.mean(pred_fake**2)

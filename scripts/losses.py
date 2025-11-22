import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    denom = pred_flat.sum(1) + target_flat.sum(1)
    loss = 1 - ((2 * intersection + smooth) / (denom + smooth))
    return loss.mean()

bce = nn.BCEWithLogitsLoss()

def bce_dice_loss(pred, target, bce_w=1.0, dice_w=1.0):
    return bce(pred, target) * bce_w + dice_loss(pred, target) * dice_w

def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1.0):
    pred = torch.sigmoid(pred)
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()
    T = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return 1 - T

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
    tv_loss = tversky_loss(pred, target, alpha, beta, smooth)
    return tv_loss ** gamma

def sobel_edges(x):
    device = x.device
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.abs(gx) + torch.abs(gy)

def edge_alignment_loss(pred, target):
    pred_p = torch.sigmoid(pred)
    e_pred = sobel_edges(pred_p)
    e_target = sobel_edges(target)
    return torch.mean(torch.abs(e_pred - e_target))

def boundary_loss(pred_logits, target):
    device = pred_logits.device
    pred_prob = torch.sigmoid(pred_logits).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch = pred_prob.shape[0]
    loss = 0.0
    for i in range(batch):
        t_edge = ndimage.distance_transform_edt(1 - target_np[i, 0])
        loss += (pred_prob[i, 0] * t_edge).mean()
    return torch.tensor(loss / batch, dtype=torch.float32, device=device)

class DUCKLoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0, tversky_w=0.7, focal_tv_w=0.7, edge_w=0.2, use_edge=True):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.tversky_w = tversky_w
        self.focal_tv_w = focal_tv_w
        self.edge_w = edge_w
        self.use_edge = use_edge

    def forward(self, pred, target):
        loss_bce = bce(pred, target)
        loss_dice = dice_loss(pred, target)
        loss_tv = tversky_loss(pred, target)
        loss_ftv = focal_tversky_loss(pred, target)
        total = self.bce_w * loss_bce + self.dice_w * loss_dice + self.tversky_w * loss_tv + self.focal_tv_w * loss_ftv
        if self.use_edge:
            total += self.edge_w * edge_alignment_loss(pred, target)
        return total

def generator_adversarial_loss(pred_fake):
    return torch.mean((pred_fake - 1.) ** 2)

def discriminator_adversarial_loss(pred_real, pred_fake):
    return torch.mean((pred_real - 1.) ** 2) + torch.mean(pred_fake ** 2)

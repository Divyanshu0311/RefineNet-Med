import torch
import numpy as np
import matplotlib.pyplot as plt

def dice_coeff(pred, target, thr=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > thr).float()
    pred = pred.view(pred.size(0), -1).cpu().numpy()
    target = target.view(target.size(0), -1).cpu().numpy()
    dices = []
    for i in range(pred.shape[0]):
        p, t = pred[i], target[i]
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append((2. * inter) / (denom + 1e-8))
    return np.mean(dices)

def save_vis(img, mask_gt, mask_pred, path):
    img = img.cpu().numpy().transpose(1,2,0)
    gt = mask_gt.cpu().numpy().squeeze()
    pr = mask_pred.cpu().numpy().squeeze()
    fig, axes = plt.subplots(1,3,figsize=(9,3))
    axes[0].imshow(img); axes[1].imshow(gt, cmap='gray'); axes[2].imshow(pr, cmap='gray')
    for a, t in zip(axes, ['Image','GT','Pred']): a.set_title(t); a.axis('off')
    plt.tight_layout(); plt.savefig(path); plt.close()

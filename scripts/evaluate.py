import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image

from models import UNet, Refiner
from dataset import KvasirSegDataset
from utils import dice_coeff, save_vis

# -------------------------------
# Metric calculations
# -------------------------------

threshold = 0.6
def compute_metrics(preds, masks):
    preds = (torch.sigmoid(preds) > threshold).float().cpu().numpy()
    masks = masks.cpu().numpy()

    dices, ious, precs, recs, f1s = [], [], [], [], []
    for p, t in zip(preds, masks):
        p = p[0]; t = t[0]
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        dice = (2 * inter) / (p.sum() + t.sum() + 1e-8)
        iou = inter / (union + 1e-8)
        prec = precision_score(t.flatten(), p.flatten(), zero_division=0)
        rec = recall_score(t.flatten(), p.flatten(), zero_division=0)
        f1 = f1_score(t.flatten(), p.flatten(), zero_division=0)
        dices.append(dice); ious.append(iou); precs.append(prec); recs.append(rec); f1s.append(f1)

    return dict(
        Dice=np.mean(dices),
        IoU=np.mean(ious),
        Precision=np.mean(precs),
        Recall=np.mean(recs),
        F1=np.mean(f1s),
    )

# -------------------------------
# Evaluation function
# -------------------------------
def evaluate_model(args, device, use_refiner=False):
    os.makedirs(args["save_dir"], exist_ok=True)
    ds = KvasirSegDataset(args["images_dir"], args["masks_dir"], img_size=args["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # load model
    model = UNet(in_channels=3, out_channels=1, base_filters=args["base_filters"]).to(device)
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model.eval()

    if use_refiner:
        print("Using Refiner for evaluation.")
        R = Refiner(in_channels=4, out_channels=1, base_filters=args["base_filters"]//2).to(device)
        R.load_state_dict(torch.load(args["refiner_path"], map_location=device))
        R.eval()
    else:
        R = None

    metrics_all = []
    out_vis_dir = os.path.join(args["save_dir"], "vis")
    os.makedirs(out_vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(loader, desc="Evaluating")):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            if use_refiner and R is not None:
                pred = torch.sigmoid(pred)
                refined = R(torch.cat([img, pred], dim=1))
                pred = refined

            metrics = compute_metrics(pred, mask)
            metrics_all.append(metrics)

            if i < 10:  # save first 10 visuals
                save_vis(img[0].cpu(), mask[0].cpu(), torch.sigmoid(pred[0].cpu()), os.path.join(out_vis_dir, f"sample_{i}.png"))

    # Aggregate results
    keys = metrics_all[0].keys()
    avg_metrics = {k: np.mean([m[k] for m in metrics_all]) for k in keys}

    print("\n📊 Evaluation Summary")
    for k, v in avg_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(metrics_all)
    df.to_csv(os.path.join(args["save_dir"], "evaluation_metrics.csv"), index=False)
    print(f"\nResults saved to: {args['save_dir']}")
    print("Visualizations saved in:", out_vis_dir)
    return avg_metrics

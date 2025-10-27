import os, random, numpy as np, torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from models import UNet, Refiner, PatchDiscriminator
from dataset import KvasirSegDataset
from losses import (
    bce_dice_loss,
    boundary_loss,
    generator_adversarial_loss,
    discriminator_adversarial_loss,
)
from utils import dice_coeff, save_vis


# ---- Reproducibility ----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- Supervised training (Stage 1) ----
def train_supervised(args, device):
    print("\n🩺 Stage 1: Supervised U-Net Training")
    set_seed()

    dataset = KvasirSegDataset(args["images_dir"], args["masks_dir"], img_size=args["img_size"], augment=True)
    n_total = len(dataset)
    val_size = max(1, int(0.15 * n_total))
    train_size = n_total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1, base_filters=args["base_filters"]).to(device)
    opt = optim.Adam(model.parameters(), lr=args["lr"])

    best_dice = 0.0
    for epoch in range(args["sup_epochs"]):
        model.train()
        train_losses = []

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['sup_epochs']}"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # ---- Validation ----
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_dices.append(dice_coeff(preds, masks))
        mean_dice = np.mean(val_dices)
        print(f"Epoch {epoch+1}: Train Loss {np.mean(train_losses):.4f} | Val Dice {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), os.path.join(args["save_dir"], "unet_pretrained.pth"))

    print("✅ Supervised pretraining done.")
    return os.path.join(args["save_dir"], "unet_pretrained.pth")


# ---- Semi-supervised + GAN refinement (Stage 2) ----
def train_semi_adversarial(args, device, pretrained_path):
    print("\n🎭 Stage 2: Adversarial Refinement (U-Net + GAN)")

    dataset = KvasirSegDataset(args["images_dir"], args["masks_dir"], img_size=args["img_size"],augment=True)
    n_total = len(dataset)
    labeled_n = max(1, int(args["label_frac"] * n_total))
    unlabeled_n = n_total - labeled_n
    labeled_ds, unlabeled_ds = random_split(dataset, [labeled_n, unlabeled_n])

    labeled_loader = DataLoader(labeled_ds, batch_size=args["batch_size"], shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args["batch_size"], shuffle=True, num_workers=2)

    # Models
    G = UNet(in_channels=3, out_channels=1, base_filters=args["base_filters"]).to(device)
    R = Refiner(in_channels=4, out_channels=1, base_filters=args["base_filters"] // 2).to(device)
    D = PatchDiscriminator(in_channels=4, base_filters=args["base_filters"] // 2).to(device)

    G.load_state_dict(torch.load(pretrained_path, map_location=device))

    optG = optim.Adam(list(G.parameters()) + list(R.parameters()), lr=args["lr"])
    optD = optim.Adam(D.parameters(), lr=args["lr"] * 0.5)

    iters = 0
    for epoch in range(args["semi_epochs"]):
        G.train()
        R.train()
        D.train()
        unlab_iter = iter(unlabeled_loader)

        for imgs_lab, masks_lab in tqdm(labeled_loader, desc=f"Semi Epoch {epoch+1}/{args['semi_epochs']}"):
            imgs_lab, masks_lab = imgs_lab.to(device), masks_lab.to(device)

            # Unlabeled batch
            try:
                imgs_un, _ = next(unlab_iter)
            except StopIteration:
                unlab_iter = iter(unlabeled_loader)
                imgs_un, _ = next(unlab_iter)
            imgs_un = imgs_un.to(device)

            # ---- Generator forward ----
            coarse_logits_lab = G(imgs_lab)
            coarse_mask_lab = torch.sigmoid(coarse_logits_lab)
            refined_logits_lab = R(torch.cat([imgs_lab, coarse_mask_lab], dim=1))
            refined_mask_lab = torch.sigmoid(refined_logits_lab)

            coarse_logits_un = G(imgs_un)
            coarse_mask_un = torch.sigmoid(coarse_logits_un)
            refined_logits_un = R(torch.cat([imgs_un, coarse_mask_un], dim=1))
            refined_mask_un = torch.sigmoid(refined_logits_un)

            # ---- Discriminator ----
            D.zero_grad()
            real_pair = torch.cat([imgs_lab, masks_lab], dim=1)
            fake_pair = torch.cat([imgs_lab, refined_mask_lab.detach()], dim=1)
            pred_real = D(real_pair)
            pred_fake = D(fake_pair)
            lossD = discriminator_adversarial_loss(pred_real, pred_fake)
            lossD.backward()
            optD.step()

            # ---- Generator (G + R) ----
            G.zero_grad()
            R.zero_grad()
            loss_sup = bce_dice_loss(refined_logits_lab, masks_lab)
            loss_boundary = boundary_loss(refined_logits_lab, masks_lab)

            pred_fake_lab = D(torch.cat([imgs_lab, refined_mask_lab], dim=1))
            loss_adv_lab = generator_adversarial_loss(pred_fake_lab)
            pred_fake_un = D(torch.cat([imgs_un, refined_mask_un], dim=1))
            loss_adv_un = generator_adversarial_loss(pred_fake_un)

            lossG = (
                args["lambda_sup"] * loss_sup
                + args["lambda_adv"] * (loss_adv_lab + loss_adv_un)
                + args["lambda_bnd"] * loss_boundary
            )
            lossG.backward()
            optG.step()

            if iters % args["log_every"] == 0:
                print(
                    f"[Iter {iters}] "
                    f"D: {lossD.item():.3f}, G: {lossG.item():.3f}, "
                    f"Sup: {loss_sup.item():.3f}, Adv: {loss_adv_lab.item():.3f}, Bnd: {loss_boundary.item():.3f}"
                )

            if iters % args["ckpt_every"] == 0:
                torch.save(G.state_dict(), os.path.join(args["save_dir"], f"G_iter{iters}.pth"))
                torch.save(R.state_dict(), os.path.join(args["save_dir"], f"R_iter{iters}.pth"))
                torch.save(D.state_dict(), os.path.join(args["save_dir"], f"D_iter{iters}.pth"))

                with torch.no_grad():
                    vis_img = imgs_lab[0].cpu()
                    vis_gt = masks_lab[0].cpu()
                    vis_pred = refined_mask_lab[0].cpu()
                    save_vis(vis_img, vis_gt, vis_pred, os.path.join(args["save_dir"], f"vis_{iters}.png"))

            iters += 1

    print("✅ Semi-supervised adversarial training complete.")

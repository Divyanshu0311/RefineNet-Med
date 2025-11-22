import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.train import train_supervised, train_semi_adversarial

def main():
    args = dict(
        images_dir="Kvasir-SEG/train/images",
        masks_dir="Kvasir-SEG/train/masks",
        save_dir="outputs/hybrid_with_duckloss",
        img_size=256,
        batch_size=4,
        base_filters=32,
        lr=1e-4,
        sup_epochs=200,
        semi_epochs=300,
        label_frac=0.3,
        lambda_sup=1.0,
        lambda_adv=0.05,
        lambda_bnd=0.5,
        log_every=20,
        ckpt_every=500
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args["save_dir"], exist_ok=True)
    print("Starting training...")
    print("Using Device:", device)
    pretrained = train_supervised(args, device)
    train_semi_adversarial(args, device, pretrained)
    print("Training finished!")

if __name__ == "__main__":
    main()
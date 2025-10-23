import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.train import train_supervised, train_semi_adversarial

def main():
    args = dict(
        images_dir="data/Kvasir-SEG/images",
        masks_dir="data/Kvasir-SEG/masks",
        save_dir="outputs",

        # ---- GPU & image configs ----
        img_size=256,           # 512 may also fit on 16GB, but 256 is safer & faster
        batch_size=8,           # 8–12 works fine for most 16GB cards (was 2)
        base_filters=48,        # 32 is balanced; 64 is heavier
        num_workers=4,          # for DataLoader parallelism

        # ---- Optimization ----
        lr=1e-4,
        sup_epochs=20,          # more stable pretraining
        semi_epochs=30,
        label_frac=0.3,
        lambda_sup=1.0,
        lambda_adv=0.05,
        lambda_bnd=0.5,
        log_every=50,
        ckpt_every=500,

        # ---- Mixed precision & memory ----
        use_amp=True            # enable automatic mixed precision in train.py
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args["save_dir"], exist_ok=True)

    print("🚀 Starting training...")
    print(f"Using device: {device}")
    print(f"CUDA Memory Total: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

    pretrained = train_supervised(args, device)
    train_semi_adversarial(args, device, pretrained)

    print("✅ Training finished successfully!")

# --- Safe entry point for multiprocessing (Windows/Linux) ---
if __name__ == "__main__":
    main()

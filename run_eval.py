import os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.evaluate import evaluate_model

def main():
    args = dict(
        images_dir="data/Kvasir-SEG/images",
        masks_dir="data/Kvasir-SEG/masks",
        # model_path="outputs/unet_pretrained.pth",  
        model_path="outputs/G_iter2200.pth",
        refiner_path="outputs/R_iter2200.pth",
        save_dir="eval_outputs/eval_results",
        img_size=256,
        base_filters=16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("🚀 Starting evaluation...")
    evaluate_model(args, device, use_refiner=True)

if __name__ == "__main__":
    main()

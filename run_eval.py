import os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.evaluate import evaluate_model

def main():
    args = dict(
        images_dir="Kvasir-SEG/test/images",
        masks_dir="Kvasir-SEG/test/masks",#"data/Kvasir-SEG/masks",
        # model_path="outputs/unet_pretrained.pth",
        model_path="new_model/G_iter11500.pth",
        refiner_path="new_model/R_iter11500.pth",
        save_dir="eval_outputs/eval_results_augmented_pinn",
        img_size=256,
        base_filters=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Starting evaluation...")
    evaluate_model(args, device, use_refiner=True)

if __name__ == "__main__":
    main()

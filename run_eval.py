import os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.evaluate import evaluate_model

def main():
    args = dict(
        images_dir="Kvasir-SEG/test/images",
        masks_dir="Kvasir-SEG/test/masks",#"data/Kvasir-SEG/masks",
        # model_path="outputs/unet_pretrained.pth",
        model_path="output_models/G_iter800.pth",
        refiner_path="output_models/R_iter800.pth",
        save_dir="eval_outputs/eval_results_test_60",
        img_size=256,
        base_filters=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("🚀 Starting evaluation...")
    evaluate_model(args, device, use_refiner=True)

if __name__ == "__main__":
    main()

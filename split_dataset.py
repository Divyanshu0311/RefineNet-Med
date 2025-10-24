import os
import shutil
import random
from tqdm import tqdm

# ---------- CONFIG ----------
DATASET_DIR = "data/Kvasir-SEG"   # root folder containing "images" and "masks"
OUTPUT_DIR = "Kvasir-SEG/"           # where split data will be saved
SPLIT_RATIO = 0.8             # 80% train, 20% test
SEED = 42                     # for reproducibility

# ---------- SETUP ----------
random.seed(SEED)
images_dir = os.path.join(DATASET_DIR, "images")
masks_dir = os.path.join(DATASET_DIR, "masks")

image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Total images found: {len(image_files)}")

# Shuffle
random.shuffle(image_files)

# Split
split_idx = int(len(image_files) * SPLIT_RATIO)
train_files = image_files[:split_idx]
test_files  = image_files[split_idx:]

# Create output directories
for subset in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, subset, "masks"), exist_ok=True)

# Copy files
def copy_pairs(file_list, subset):
    for fname in tqdm(file_list, desc=f"Copying {subset} data"):
        img_src = os.path.join(images_dir, fname)
        mask_src = os.path.join(masks_dir, fname)
        img_dst = os.path.join(OUTPUT_DIR, subset, "images", fname)
        mask_dst = os.path.join(OUTPUT_DIR, subset, "masks", fname)

        # Check if mask exists
        if not os.path.exists(mask_src):
            print(f"⚠️  Warning: Mask not found for {fname}, skipping.")
            continue

        shutil.copy2(img_src, img_dst)
        shutil.copy2(mask_src, mask_dst)

copy_pairs(train_files, "train")
copy_pairs(test_files, "test")

print("\n✅ Dataset split completed!")
print(f"Train samples: {len(train_files)}")
print(f"Test samples:  {len(test_files)}")
print(f"Saved under '{OUTPUT_DIR}/train' and '{OUTPUT_DIR}/test'")

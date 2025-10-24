import os, sys, torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from scripts.models import UNet, Refiner
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

IMG_PATH   = "test/images/10.png"         
MASK_PATH  = "test/masks/10.png"     
GEN_PATH   = "models/20x48x256/G_iter1000.pth"          # pretrained U-Net generator
# GEN_PATH   = "outputs/G_iter2200.pth"          # pretrained U-Net generator
# REF_PATH   = "outputs/R_iter2200.pth"          # trained refiner
REF_PATH   = "models/20x48x256/R_iter1000.pth"          # trained refiner
IMG_SIZE   = 256                               # must match training
BASE_FILTERS = 48                             # same as training
THRESHOLD = 0.90                             # mask threshold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

img = Image.open(IMG_PATH).convert("RGB")
mask_gt = Image.open(MASK_PATH).convert("L")
tf_img = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()])
tf_mask = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

x = tf_img(img).unsqueeze(0).to(device)
y_gt = tf_mask(mask_gt)[0].cpu().numpy()


unet = UNet(in_channels=3, out_channels=1, base_filters=BASE_FILTERS)
unet.load_state_dict(torch.load(GEN_PATH, map_location=device))
unet.to(device).eval()

refiner = Refiner(in_channels=4, out_channels=1, base_filters=BASE_FILTERS//2)
refiner.load_state_dict(torch.load(REF_PATH, map_location=device))
refiner.to(device).eval()

with torch.no_grad():
    coarse = torch.sigmoid(unet(x))
    ref_input = torch.cat([x, coarse], dim=1)
    refined = torch.sigmoid(refiner(ref_input))
    pred_map = refined[0, 0].cpu().numpy()             # raw probability map
    pred_mask = (pred_map > THRESHOLD).astype(np.uint8)  # binary mask

plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
plt.imshow(np.array(img))
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(y_gt, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(pred_map, cmap="gray")   # continuous heatmap
plt.title("Predicted Probabilities")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(pred_mask, cmap="gray")
plt.title(f"Thresholded Mask (>{THRESHOLD})")
plt.axis("off")

plt.tight_layout()
plt.show()
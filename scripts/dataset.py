import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class KvasirSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.jpg') or f.endswith('.png')])
        assert len(self.images) == len(self.masks)
        self.tf_img = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop((img_size // 16 * 16, img_size // 16 * 16)),
            T.ToTensor(),
        ])
        self.tf_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop((img_size // 16 * 16, img_size // 16 * 16)),
            T.ToTensor(),
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.tf_img(Image.open(self.images[idx]).convert('RGB'))
        mask = self.tf_mask(Image.open(self.masks[idx]).convert('L'))
        mask = (mask > 0.5).float()
        return img, mask

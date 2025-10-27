import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KvasirSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256, augment=False):
        self.images = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        self.masks = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"
        self.augment = augment
        self.img_size = img_size

        # --- Augmentation pipeline for training ---
        self.train_tf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.CLAHE(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.5),
            A.GaussianBlur(3, p=0.2),
            ToTensorV2()
        ])

        # --- Only resize for validation/testing ---
        self.val_tf = A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]).convert('L'), dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        mask = (mask > 127).astype(np.float32)  # binarize mask if needed

        if self.augment:
            transformed = self.train_tf(image=img, mask=mask)
        else:
            transformed = self.val_tf(image=img, mask=mask)

        img_tensor = transformed["image"]
        mask_tensor = transformed["mask"].permute(2, 0, 1)
        return img_tensor, mask_tensor

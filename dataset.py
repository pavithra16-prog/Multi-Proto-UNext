import os
import cv2
import numpy as np
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext='.jpg', mask_ext='.png', transform=None):
        """
        Args:
            img_ids (list): List of image IDs (without extension).
            img_dir (str): Directory containing input images.
            mask_dir (str): Directory containing masks. Should have '0' folder for binary masks.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (albumentations.Compose, optional): Optional data augmentation.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load image
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load mask (binary)
        mask_path = os.path.join(self.mask_dir, '0', img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Convert mask to binary 0/1 integers
        mask = np.where(mask > 127, 1, 0).astype(np.uint8)  # ensures 0/1, dtype=uint8
        mask = mask[..., None]  # shape: (H, W, 1)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert image to float and CHW format
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (C, H, W)
        mask = mask.transpose(2, 0, 1)  # (1, H, W)

        # Convert to torch tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()  # long for integer class labels

        return img, mask, {'img_id': img_id}

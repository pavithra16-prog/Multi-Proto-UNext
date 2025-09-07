import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from proto_model import UNeXtWithPrototypes  # <-- Prototype-integrated UNeXt
from albumentations import Compose, Resize, Normalize
from torch.utils.data import DataLoader

# === CONFIG ===
model_name = 'BUSI_UNeXt_proto_num1_seed43'  # Change as needed
model_path = f'/home/gpavithra/AIP/UNeXtMultiProto-pytorch/models/{model_name}/model.pth'
test_ids_file = '/home/gpavithra/AIP/UNeXtMultiProto-pytorch/test_ids.txt'
output_dir = f'/home/gpavithra/AIP/UNeXtMultiProto-pytorch/outputs1/{model_name}/vis'
img_dir = '/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs/BUSI/images'
mask_dir = '/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs/BUSI/masks/0'
img_ext = '.png'
mask_ext = '.png'
input_h, input_w = 256, 256
input_channels = 3
num_classes = 1
base_c = 32
proto_dim = 16
num_prototypes = 1

os.makedirs(output_dir, exist_ok=True)

# === Load test IDs ===
with open(test_ids_file, 'r') as f:
    test_ids = [line.strip() for line in f.readlines()]

# === Albumentations Transform ===
transform = Compose([
    Resize(input_h, input_w),
    Normalize(),
])

# === Dataset Class ===
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, img_dir, mask_dir, img_ext, mask_ext, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)

        img_orig = cv2.imread(img_path)  # BGR
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_w, input_h))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]
        mask = cv2.resize(mask, (input_w, input_h))

        augmented = self.transform(image=img_resized, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        mask = np.expand_dims(mask, axis=-1)
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).float() / 255.0

        return img_id, img, mask, img_resized  # img_resized is original RGB resized

# === Load Prototype-Integrated Model ===
model = UNeXtWithPrototypes(in_channels=input_channels, num_classes=num_classes, base_c=base_c, proto_dim=proto_dim, num_prototypes=num_prototypes)
# model.load_state_dict(torch.load(model_path))

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

# === DataLoader ===
test_dataset = TestDataset(test_ids, img_dir, mask_dir, img_ext, mask_ext, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === Inference & Visualization ===
print(f"Saving visualisations to {output_dir}")
for img_id, img, gt_mask, orig_img_rgb in tqdm(test_loader):
    img = img.cuda()
    with torch.no_grad():
        # logits = model(img)  # No need to unpack features here
        # pred_mask = torch.sigmoid(logits)
        logits, _, _ = model(img)
        pred_mask = torch.sigmoid(logits)
        pred_mask = (pred_mask > 0.5).float()

    # Prepare visualization
    gt_np = gt_mask[0].cpu().numpy().squeeze() * 255
    pred_np = pred_mask[0][0].cpu().numpy() * 255
    vis_gt = gt_np.astype(np.uint8)
    vis_pred = pred_np.astype(np.uint8)
    # vis_orig = orig_img_rgb[0].astype(np.uint8)
    vis_orig = orig_img_rgb[0].cpu().numpy().astype(np.uint8)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vis_orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(vis_gt, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(vis_pred, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{img_id[0]}_vis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

print("Done! Multi-Proto-UNeXt visualisations saved.")

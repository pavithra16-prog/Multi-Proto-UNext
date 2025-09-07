# test.py
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import Dataset
from metrics import iou_score
from proto_model import UNeXtWithPrototypes  # <-- Updated import
import albumentations as A
from albumentations import Resize, Normalize

def load_config(model_name):
    with open(f'/home/gpavithra/AIP/UNeXtMultiProto-pytorch/models/{model_name}/config.yml') as f:
        return yaml.safe_load(f)

def load_test_ids():
    with open("/home/gpavithra/AIP/UNeXtMultiProto-pytorch/test_ids.txt") as f:
        return [line.strip() for line in f.readlines()]

def evaluate_model(model_name):
    config = load_config(model_name)
    test_ids = load_test_ids()

    # Load the prototype-integrated UNeXt model
    model = UNeXtWithPrototypes(
        in_channels=config['input_channels'],
        num_classes=config['num_classes'],
        base_c=config.get('base_c', None),      # Optional: can pass None or config value
        proto_dim=config.get('proto_dim', None), # Optional: will default to feature dim if None
        num_prototypes=config.get('num_prototypes', 1) 
            )

    # model.load_state_dict(torch.load(f'/home/gpavithra/AIP/UNeXtMultiProto-pytorch/models/{model_name}/model.pth'))

    checkpoint = torch.load(f'/home/gpavithra/AIP/UNeXtMultiProto-pytorch/models/{model_name}/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.cuda().eval()

    # Transforms
    test_transform = A.Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize()
    ])

    img_dir = f"/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs/{config['dataset']}/images"
    mask_dir = f"/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs/{config['dataset']}/masks/"
    
    print("Image Directory:", img_dir)
    print("Mask Directory:", mask_dir)
    
    # Dataset
    test_dataset = Dataset(
        img_ids=test_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    ious = []
    dices = []

    with torch.no_grad():
        for input, target, _ in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            input = input.cuda()
            target = target.cuda()

            logits, _, _ = model(input)  # <-- only use logits
            iou, dice = iou_score(logits, target)
            ious.append(iou)
            dices.append(dice)

    ious = np.array(ious)
    dices = np.array(dices)

    return {
        "iou_mean": np.mean(ious),
        "iou_std": np.std(ious),
        "dice_mean": np.mean(dices),
        "dice_std": np.std(dices),
    }

if __name__ == "__main__":
    models = ["BUSI_UNeXt_proto_num6_seed42", "BUSI_UNeXt_proto_num6_seed43", "BUSI_UNeXt_proto_num6_seed44"]
    all_results = []

    for m in models:
        result = evaluate_model(m)
        all_results.append(result)
        print(f"\n{m} → IoU: {result['iou_mean']:.4f}, Dice: {result['dice_mean']:.4f}")

    # Final summary
    print("\n=== Final Aggregated Results ===")
    ious = [r["iou_mean"] for r in all_results]
    dices = [r["dice_mean"] for r in all_results]
    print(f"Mean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")

import argparse
import os
from collections import OrderedDict
from glob import glob
import random

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

# Albumentations
import albumentations as A
from albumentations import RandomRotate90, Resize, HorizontalFlip, VerticalFlip
from albumentations.core.composition import Compose

# Local modules
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

from proto_model import UNeXtWithPrototypes
from losses import compute_prototype_loss

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--proto_loss_weight', type=float, default=0.1)
    parser.add_argument('--base_c', type=int, default=32)
    parser.add_argument('--proto_dim', type=int, default=64)
    parser.add_argument('--use_prototypes', default=False, type=str2bool)
    parser.add_argument('--num_prototypes', default=1, type=int)

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNeXt')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--dataset', default='isic')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N')
    parser.add_argument('--cfg', type=str, metavar="FILE")

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train(config, train_loader, model, criterion, optimizer, lambda_proto):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader), desc="Training", ncols=120)
    for input, target, _ in train_loader:
        input, target = input.cuda(), target.cuda().float()

        # forward
        logits, features, prototypes = model(input)
        seg_loss = criterion(logits, target)
        B, C, H, W = features.shape
        features = features.permute(0,2,3,1).reshape(-1, C)
        targets_flat = target.view(-1).long()
        prototypes = prototypes.view(-1, features.shape[1])  # shape: (num_classes*num_prototypes, C)
        proto_loss = compute_prototype_loss(features, targets_flat, prototypes, config["num_prototypes"])
        # proto_loss = compute_prototype_loss(features, targets_flat, prototypes, config["num_prototypes"])
        total_loss = seg_loss + lambda_proto * proto_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # metrics
        iou, dice = iou_score(logits, target)
        avg_meters['loss'].update(seg_loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        pbar.set_postfix(OrderedDict([
            ('λ_proto', lambda_proto),
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('p_loss', proto_loss.item())
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, lambda_proto):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validating", ncols=120)
        for input, target, _ in val_loader:
            input, target = input.cuda(), target.cuda().float()

            logits, features, prototypes = model(input)
            seg_loss = criterion(logits, target)
            B, C, H, W = features.shape
            features = features.permute(0,2,3,1).reshape(-1, C)
            targets_flat = target.view(-1).long()
            prototypes = prototypes.view(-1, features.shape[1])  # shape: (num_classes*num_prototypes, C)
            proto_loss = compute_prototype_loss(features, targets_flat, prototypes, config["num_prototypes"])
            # proto_loss = compute_prototype_loss(features, targets_flat, prototypes, config["num_prototypes"])
            loss = seg_loss + lambda_proto * proto_loss

            iou, dice = iou_score(logits, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            pbar.set_postfix(OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ]))
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def main():
    args = parse_args()
    config = vars(args)
    set_seed(config['seed'])
    print("CUDA available:", torch.cuda.is_available())

    if config['name'] is None:
        config['name'] = f"{config['dataset']}_{config['arch']}" + ("_wDS" if config['deep_supervision'] else "_woDS")

    save_dir = f"/home/gpavithra/AIP/UNeXtMultiProto-pytorch/models/{config['name']}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # criterion
    criterion = nn.BCEWithLogitsLoss().cuda() if config['loss'] == 'BCEWithLogitsLoss' else losses.__dict__[config['loss']]().cuda()
    cudnn.benchmark = True

    # model
    if config.get("use_prototypes", False):
        model = UNeXtWithPrototypes(
            in_channels=config['input_channels'],
            num_classes=config['num_classes'],
            base_c=config['base_c'],
            proto_dim=config['proto_dim'],
            num_prototypes=config['num_prototypes']
        )
    else:
        model = archs.UNeXt(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config['deep_supervision']
        )
    model = model.cuda()

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay']) \
        if config['optimizer'] == 'Adam' else optim.SGD(
            params, lr=config['lr'], momentum=config['momentum'],
            nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    # scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'],
                                                   patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    else:
        scheduler = None

    # dataset split
    img_ids = glob(os.path.join('/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs', config['dataset'], 'images', f'*{config["img_ext"]}'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    with open(config.get('test_ids_path', '/home/gpavithra/AIP/UNeXtMultiProto-pytorch/test_ids.txt'), 'r') as f:
        test_ids = [line.strip() for line in f]
    trainval_ids = [i for i in img_ids if i not in test_ids]
    train_ids, val_ids = train_test_split(trainval_ids, test_size=0.2, random_state=config['seed'])

    train_transform = Compose([RandomRotate90(p=0.5), HorizontalFlip(p=0.5),
                               Resize(config['input_h'], config['input_w']), A.Normalize()])
    val_transform = Compose([Resize(config['input_h'], config['input_w']), A.Normalize()])

    train_dataset = Dataset(train_ids,
                            os.path.join('/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs', config['dataset'], 'images'),
                            os.path.join('/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs', config['dataset'], 'masks'),
                            config['img_ext'], config['mask_ext'], train_transform)

    val_dataset = Dataset(val_ids,
                          os.path.join('/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs', config['dataset'], 'images'),
                          os.path.join('/home/gpavithra/AIP/UNeXtMultiProto-pytorch/inputs', config['dataset'], 'masks'),
                          config['img_ext'], config['mask_ext'], val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'], drop_last=False)

    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('iou', []), ('val_loss', []), ('val_iou', []), ('val_dice', [])])
    best_iou, trigger, lambda_proto, start_epoch = 0, 0, config['proto_loss_weight'], 0

    # resume
    if args.resume is not None:
        print(f"=> Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"=> Resumed from epoch {checkpoint.get('epoch', 0)}")

    for epoch in range(start_epoch, config['epochs']):
        print(f'Epoch [{epoch+1}/{config["epochs"]}]')
        train_log = train(config, train_loader, model, criterion, optimizer, lambda_proto)
        val_log = validate(config, val_loader, model, criterion, lambda_proto)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(f'train_loss {train_log["loss"]:.4f} train_iou {train_log["iou"]:.4f} '
              f' val_loss {val_log["loss"]:.4f} val_iou {val_log["iou"]:.4f}')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv(f'{save_dir}/log.csv', index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            checkpoint_path = f'{save_dir}/model.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        if (epoch + 1) % 50 == 0:
            checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print(f"=> saved checkpoint at epoch {epoch+1}")

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

# -*- coding: UTF-8 -*-

import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

from data_loader.GetDataset_DRIVE import randomHorizontalFlip, randomHueSaturationValue, randomVerticleFlip


def default_deepcrack_loader(
    img_path,
    mask_path,
    binary_mask_path=None,
    resize_shape=(512, 512),
    train=False,
    label_mode='binary',
):
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_shape)

    if label_mode == 'binary':
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = mask.copy()
    elif label_mode in ('dist', 'dist_inverted'):
        mask = np.load(mask_path, allow_pickle=True)
        binary_mask = (
            cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
            if binary_mask_path is not None
            else np.zeros_like(mask)
        )

    interp = cv2.INTER_NEAREST if label_mode == 'binary' else cv2.INTER_LINEAR
    mask = cv2.resize(mask, resize_shape, interpolation=interp)
    binary_mask = cv2.resize(binary_mask, resize_shape, interpolation=cv2.INTER_NEAREST)

    if train:
        # brightness / contrast jitter (HSV)
        img = randomHueSaturationValue(
            img,
            hue_shift_limit=(-30, 30),
            sat_shift_limit=(-5, 5),
            val_shift_limit=(-15, 15),
        )

        # discrete rotation: uniform from {0°, 90°, 180°, 270°}
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
            binary_mask = np.rot90(binary_mask, k=k).copy()

        # flip (both masks transformed together for consistency)
        combined = np.stack([mask, binary_mask], axis=-1)
        img, combined = randomHorizontalFlip(img, combined)
        img, combined = randomVerticleFlip(img, combined)
        mask = combined[..., 0]
        binary_mask = combined[..., 1]

    mask = np.expand_dims(mask, axis=2)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    binary_mask = np.array(binary_mask, np.float32).transpose(2, 0, 1)

    if label_mode == 'binary':
        mask = mask / 255.0
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

    binary_mask = binary_mask / 255.0
    binary_mask[binary_mask >= 0.5] = 1
    binary_mask[binary_mask < 0.5] = 0

    return img, mask, binary_mask


class MyDataset_DeepCrack(data.Dataset):
    def __init__(self, args, train_root, mode='train', label_mode='binary'):
        self.args = args
        self.label_mode = label_mode
        self.train = (mode == 'train')

        img_dir = f"{train_root}/{mode}_img/"
        lab_dir = f"{train_root}/{mode}_lab/"

        if label_mode == 'binary':
            label_postfix = '.png'
        elif label_mode == 'dist':
            lab_dir = f"{train_root}/{mode}_lab_dist/"
            label_postfix = '_dist.npy'
        elif label_mode == 'dist_inverted':
            lab_dir = f"{train_root}/{mode}_lab_dist_inverted/"
            label_postfix = '_dist_inverted.npy'

        binary_lab_dir = f"{train_root}/{mode}_lab/"

        img_ls, mask_ls, binary_mask_ls, name_ls = [], [], [], []
        for img_path in sorted(glob.glob(img_dir + '*.jpg')):
            stem = Path(img_path).stem
            img_ls.append(img_path)
            mask_ls.append(lab_dir + stem + label_postfix)
            binary_mask_ls.append(binary_lab_dir + stem + '.png')
            name_ls.append(stem)

        self.img_ls = img_ls
        self.mask_ls = mask_ls
        self.binary_mask_ls = binary_mask_ls
        self.name_ls = name_ls

    def __getitem__(self, index):
        resize_shape = tuple(self.args.resize[:2])
        binary_mask_path = self.binary_mask_ls[index] if self.label_mode in ('dist', 'dist_inverted') else None
        img, mask, binary_mask = default_deepcrack_loader(
            self.img_ls[index], self.mask_ls[index], binary_mask_path,
            resize_shape, self.train, self.label_mode,
        )
        return {
            'image': torch.Tensor(img),
            'label': torch.Tensor(mask),
            'binary_gt': torch.Tensor(binary_mask),
            'name': self.name_ls[index],
        }

    def __len__(self):
        return len(self.img_ls)

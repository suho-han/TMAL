# -*- coding: UTF-8 -*-

import glob
import math
import os
import random

import cv2
import numpy as np
import scipy.io as scio
import scipy.misc as misc
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.io import imread, imsave
from torchvision import transforms


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        # Keep upstream uint8 wraparound behavior while avoiding Python 3.13
        # OverflowError on direct negative-to-uint8 conversion.
        hue_shift = np.uint8(hue_shift % 256)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * sx
        ss = math.sin(angle / 180 * math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + \
            np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(img_path, mask_path):

    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))

    img = randomHueSaturationValue(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2, 0, 1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def default_DRIVE_loader(img_path, mask_path, binary_mask_path=None, resize_shape=(960, 960), train=False, label_mode='binary', grayscale_input=False):
    if grayscale_input:
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imread(img_path)
    img = cv2.resize(img, resize_shape)
    if label_mode == 'binary':
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = mask.copy()
    elif label_mode in ['dist', 'dist_inverted']:
        mask = np.load(mask_path, allow_pickle=True)
        if binary_mask_path is not None:
            binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            binary_mask = np.zeros_like(mask)

    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST if label_mode == 'binary' else cv2.INTER_LINEAR)
    binary_mask = cv2.resize(binary_mask, resize_shape, interpolation=cv2.INTER_NEAREST)

    if train:
        img = randomHueSaturationValue(img, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        
        # Combine masks to apply the same random transformation
        combined_mask = np.stack([mask, binary_mask], axis=-1)
        img, combined_mask = randomShiftScaleRotate(img, combined_mask, shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), aspect_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
        img, combined_mask = randomHorizontalFlip(img, combined_mask)
        img, combined_mask = randomVerticleFlip(img, combined_mask)
        img, combined_mask = randomRotate90(img, combined_mask)
        mask = combined_mask[..., 0]
        binary_mask = combined_mask[..., 1]

    mask = np.expand_dims(mask, axis=2)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    binary_mask = np.array(binary_mask, np.float32).transpose(2, 0, 1)
    
    if label_mode == 'binary':
        mask = mask / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
    elif label_mode in ['dist', 'dist_inverted']:
        pass
    
    binary_mask = binary_mask / 255.0
    binary_mask[binary_mask >= 0.5] = 1
    binary_mask[binary_mask <= 0.5] = 0

    return img, mask, binary_mask


class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image = (image-image.min())/(image.max()-image.min())
        mask = mask/255.0
        if mask is None:
            return image
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask=None):
        H, W = image.shape
        randw = np.random.randint(W/8)
        randh = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1].copy(), mask[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask


def Resize(image, mask, H, W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        return image, mask
    else:
        return image


class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)

        return image, mask


def _resize_image(image, target):
    return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_LINEAR)


class MyDataset_DRIVE(data.Dataset):
    def __init__(self, args, train_root, mode='train', label_mode='binary'):
        train = True if mode == 'train' else False
        self.args = args
        self.label_mode = label_mode
        img_path = train_root+'/'+mode+'/images/'
        gt_path = train_root+'/'+mode+'/masks/'

        img_ls = []
        mask_ls = []
        binary_mask_ls = []
        name_ls = []

        if self.label_mode == 'binary':
            label_postfix = '.gif'
        elif self.label_mode == 'dist':
            gt_path = train_root+'/'+mode+'/masks_dist/'
            label_postfix = '_dist.npy'
        elif self.label_mode == 'dist_inverted':
            gt_path = train_root+'/'+mode+'/masks_dist_inverted/'
            label_postfix = '_dist_inverted.npy'

        img_list = glob.glob(img_path+'*.tif')
        for img_id in img_list:
            img = img_path+str(img_id.split('/')[-1])
            stem = str(img_id.split('/')[-1].split('.tif')[0])
            gt = gt_path+stem+label_postfix
            binary_gt = train_root+'/'+mode+'/masks/'+stem+'.gif'
            name = stem
            img_ls.append(img)
            mask_ls.append(gt)
            binary_mask_ls.append(binary_gt)
            name_ls.append(name)

        self.train = train
        # print(file)

        self.name_ls = name_ls
        self.img_ls = img_ls

        self.mask_ls = mask_ls
        self.binary_mask_ls = binary_mask_ls

        self.normalize = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.totensor = ToTensor()

    def __getitem__(self, index):
        resize_shape = tuple(self.args.resize[:2])
        binary_mask_path = self.binary_mask_ls[index] if self.label_mode in ['dist', 'dist_inverted'] else None
        img, mask, binary_mask = default_DRIVE_loader(self.img_ls[index], self.mask_ls[index], binary_mask_path, resize_shape, self.train, self.label_mode)

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        binary_mask = torch.Tensor(binary_mask)
        
        return {
            'image': img.squeeze(0) if img.ndim == 4 else img, # wait, default_DRIVE_loader returns (3, H, W). squeeze(0) is not needed. Actually previous code did img.squeeze(0), wait!
            'label': mask,
            'binary_gt': binary_mask,
            'name': self.name_ls[index]
        }

    def __len__(self):
        return len(self.img_ls)




def connectivity_matrix(mask):
    # print(mask.shape)
    [batch, channels, rows, cols] = mask.shape

    conn = torch.ones([batch, 8, rows, cols])
    up = torch.zeros([batch, rows, cols])  # move the orignal mask to up
    down = torch.zeros([batch, rows, cols])
    left = torch.zeros([batch, rows, cols])
    right = torch.zeros([batch, rows, cols])
    up_left = torch.zeros([batch, rows, cols])
    up_right = torch.zeros([batch, rows, cols])
    down_left = torch.zeros([batch, rows, cols])
    down_right = torch.zeros([batch, rows, cols])

    up[:, :rows-1, :] = mask[:, 0, 1:rows, :]
    down[:, 1:rows, :] = mask[:, 0, 0:rows-1, :]
    left[:, :, :cols-1] = mask[:, 0, :, 1:cols]
    right[:, :, 1:cols] = mask[:, 0, :, :cols-1]
    up_left[:, 0:rows-1, 0:cols-1] = mask[:, 0, 1:rows, 1:cols]
    up_right[:, 0:rows-1, 1:cols] = mask[:, 0, 1:rows, 0:cols-1]
    down_left[:, 1:rows, 0:cols-1] = mask[:, 0, 0:rows-1, 1:cols]
    down_right[:, 1:rows, 1:cols] = mask[:, 0, 0:rows-1, 0:cols-1]

    # print(mask.shape,down_right.shape)
    conn[:, 0] = mask[:, 0]*down_right
    conn[:, 1] = mask[:, 0]*down
    conn[:, 2] = mask[:, 0]*down_left
    conn[:, 3] = mask[:, 0]*right
    conn[:, 4] = mask[:, 0]*left
    conn[:, 5] = mask[:, 0]*up_right
    conn[:, 6] = mask[:, 0]*up
    conn[:, 7] = mask[:, 0]*up_left
    conn = conn.float()

    return conn


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def check_label(mask):
    label = np.array([1, 0, 0, 0])
    # print(mask.shape)
    # print(mask[1,:,:].max())
    if mask[1, :, :].max() != 0:
        label[1] = 1

    if mask[2, :, :].max() != 0:
        label[2] = 1

    if mask[3, :, :].max() != 0:
        label[3] = 1

    return label

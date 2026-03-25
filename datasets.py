# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets

import utils


ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]

    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, title, desc = self.samples[i]
            caption = np.random.choice([title, desc])
            img = yfcc_loader(self.root, index)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc3m':
            ann = self.samples[i]
            filename, captions = ann['image_id'], ann['captions']
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc12m':
            ann = self.samples[i]
            filename, captions = ann['image_name'], ann['captions']
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment):
        super().__init__(dataset, root, metadata)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


# =============================================================================
# QuickDraw! SketchRNN dataset
# Data format: sketchrnn_{class_name}.full.npz, keys: train / valid / test
# Each value is a 1-D numpy object array of (N, 3) int16 stroke arrays.
# Stroke-3 format: (dx, dy, pen_state) — pen_state=0: down, 1: lift after point
# =============================================================================

# holdout set 1
QD_HOLDOUT_CLASSES = [
    'bicycle', 'train', 'stop sign', 'dog', 'elephant',
    'backpack', 'skateboard', 'knife', 'sandwich', 'pizza',
    'couch', 'mouse', 'oven', 'clock',
]

QD_TEMPLATES = [
    "a drawing of a {}",
    "a sketch of a {}",
    "a rough sketch of {}",
    "a doodle of a {}",
    "a hand-drawn {}",
]


def rasterize_stroke3(strokes, size=224, line_width=2, padding=10):
    """
    Convert stroke-3 format to a PIL RGB image (white background, black strokes).
    Follows official strokes_to_lines logic from googlecreativelab/quickdraw-dataset.

    pen_state=0: pen down — continue current stroke
    pen_state=1: pen up   — include this point in the current stroke, then end it;
                            next point starts a fresh stroke with no connecting line
    """
    # Reconstruct absolute coordinates via cumulative sum of relative deltas
    abs_coords = np.cumsum(strokes[:, :2], axis=0).astype(float)
    pen_states = strokes[:, 2]

    # Normalise all points to fit within the canvas bounds
    x, y = abs_coords[:, 0], abs_coords[:, 1]
    x_range = x.max() - x.min() or 1
    y_range = y.max() - y.min() or 1
    scale = (size - 2 * padding) / max(x_range, y_range)
    x = ((x - x.min()) * scale + padding).astype(int)
    y = ((y - y.min()) * scale + padding).astype(int)

    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Group points into individual strokes; draw each as a filled polyline
    stroke_points = []
    for i in range(len(strokes)):
        stroke_points.append((int(x[i]), int(y[i])))
        if pen_states[i] == 1:          # end of this stroke
            if len(stroke_points) >= 2:
                draw.line(stroke_points, fill=(0, 0, 0), width=line_width)
            stroke_points = []          # start fresh for the next stroke

    # Safety: render any trailing points in malformed sequences
    if len(stroke_points) >= 2:
        draw.line(stroke_points, fill=(0, 0, 0), width=line_width)

    return img


def get_sketchrnn_class_names(root, holdout=QD_HOLDOUT_CLASSES, train_scheme='o'):
    """Return sorted class names from SketchRNN npz directory, excluding holdout classes."""
    names = []
    for f in sorted(os.listdir(root)):
        if f.endswith('.full.npz'): # example : 'light bulb.full.npz'
            name = f[:-len('.full.npz')]
            if train_scheme=='c' or name not in holdout:
                names.append(name)
    return names


class QuickDraw(torch.utils.data.Dataset):
    """
    Base dataset for the QuickDraw SketchRNN subset.

    Args:
        root (str): Directory containing sketchrnn_*.full.npz files.
        class_names (list[str]): Class names to include (file stem without prefix/suffix).
        split (str): 'train', 'valid', or 'test'.
        transform: torchvision transform applied to rasterised PIL images.
    """

    def __init__(self, root, class_names, split='train', transform=None):
        assert split in ('train', 'valid', 'test')
        self.root = root
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.split = split
        self.transform = transform
        # True mmap: .npy files map directly to file bytes, no zip wrapper.
        # ptr is tiny (~560 KB/class), loaded fully. strokes is mmap'd —
        # only touched pages are in RAM, shared across GPU processes via OS page cache.
        self._mmap_cache = {}  # class_name -> (ptr, strokes memmap)
        self.samples = []
        for class_name in class_names:
            ptr = np.load(os.path.join(root, f'{class_name}.{split}.ptr.npy'))
            strokes = np.load(os.path.join(root, f'{class_name}.{split}.strokes.npy'), mmap_mode='r')
            self._mmap_cache[class_name] = (ptr, strokes)
            class_idx = self.class_to_idx[class_name]
            for i in range(len(ptr) - 1):
                self.samples.append((class_idx, i))

    def _load_sample(self, class_name, sample_idx):
        ptr, strokes = self._mmap_cache[class_name]
        return strokes[ptr[sample_idx]:ptr[sample_idx + 1]]

    def get_pil_image(self, class_idx, sample_idx):
        class_name = self.class_names[class_idx]
        strokes = self._load_sample(class_name, sample_idx)  # (T, 3) int16
        return rasterize_stroke3(strokes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        raise NotImplementedError


class QD_CLIP(QuickDraw):
    """Returns (image_tensor, caption_tokens) for CLIP training."""

    def __init__(self, root, class_names, split='train', transform=None, tokenizer=None):
        super().__init__(root, class_names, split=split, transform=transform)
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        class_idx, sample_idx = self.samples[i]
        img = self.get_pil_image(class_idx, sample_idx)

        if self.transform is not None:
            image = self.transform(img)

        class_name = self.class_names[class_idx]
        caption = random.choice(QD_TEMPLATES).format(class_name)
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class QD_SLIP(QuickDraw):
    """Returns (image_tensor, caption_tokens, aug1, aug2) for SLIP training."""

    def __init__(self, root, class_names, split='train', transform=None, augment=None, tokenizer=None):
        super().__init__(root, class_names, split=split, transform=transform)
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        class_idx, sample_idx = self.samples[i]
        img = self.get_pil_image(class_idx, sample_idx)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        class_name = self.class_names[class_idx]
        caption = random.choice(QD_TEMPLATES).format(class_name)
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class QD_SSL(QuickDraw):
    """Returns (aug1, aug2) for SIMCLR-only training (no captions)."""

    def __init__(self, root, class_names, split='train', augment=None):
        super().__init__(root, class_names, split=split, transform=None)
        self.augment = augment

    def __getitem__(self, i):
        class_idx, sample_idx = self.samples[i]
        img = self.get_pil_image(class_idx, sample_idx)
        return self.augment(img), self.augment(img)


class QD_Val(QuickDraw):
    """
    Returns (image_tensor, int_label) over the 14 held-out classes for zero-shot eval.
    Integer label indexes into holdout_classes list.
    """

    def __init__(self, root, holdout_classes, split='valid', transform=None):
        super().__init__(root, holdout_classes, split=split, transform=transform)

    def __getitem__(self, i):
        class_idx, sample_idx = self.samples[i]
        img = self.get_pil_image(class_idx, sample_idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, class_idx
# TODO: Test that the QD_CLIP/QD_SLIP/QD_SSL is working as intended

def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    elif entry['type'] == 'quickdraw_npz':
        dataset = QD_Val(root=root, holdout_classes=QD_HOLDOUT_CLASSES,
                         split='valid', transform=transform)
    else:
        raise Exception('Unknown dataset')

    return dataset


def get_dataset(train_transform, tokenizer, args, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if args.dataset == 'quickdraw':
        # Sketch-adapted augmentation: no ColorJitter (already greyscale) or
        # RandomGrayscale; keep crop and horizontal flip for invariance.
        sketch_augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        class_names = get_sketchrnn_class_names(args.root, train_scheme=args.train_scheme)
        if args.model.startswith('SIMCLR'):
            return QD_SSL(args.root, class_names, split='train', augment=sketch_augment)
        elif args.model.startswith('CLIP'):
            return QD_CLIP(args.root, class_names, split='train',
                           transform=train_transform, tokenizer=tokenizer)
        elif args.model.startswith('SLIP'):
            return QD_SLIP(args.root, class_names, split='train',
                           transform=train_transform, augment=sketch_augment, tokenizer=tokenizer)

    # Standard image-caption datasets (unchanged)
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.model.startswith('SIMCLR'):
        return ImageCaptionDatasetSSL(args.dataset, args.root, args.metadata, augment)
    elif args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer)
    elif args.model.startswith('SLIP'):
        return ImageCaptionDatasetSLIP(args.dataset, args.root, args.metadata, train_transform, augment, tokenizer)
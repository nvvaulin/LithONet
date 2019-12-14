import cv2
import numpy as np

from albumentations import (
    CLAHE, Blur, CoarseDropout, Compose, GaussianBlur, HorizontalFlip, InvertImg, OneOf, RandomBrightnessContrast,
    RandomRotate90, RandomSizedCrop, Resize, Rotate, ShiftScaleRotate, VerticalFlip)
from seismic.config import config


def array_freqhist_bins(img, n_bins=100):
    imsd = np.sort(img.flatten())
    t = np.array([0.001])
    t = np.append(t, np.arange(n_bins) / n_bins + (1 / 2 / n_bins))
    t = np.append(t, 0.999)
    t = (len(imsd) * t + 0.5).astype(np.int)
    return np.unique(imsd[t])


def scale_img(img, brks=None):
    if brks is None:
        brks = array_freqhist_bins(img)
    ys = np.linspace(0., 1., len(brks))
    x = np.interp(img.flatten(), brks, ys)
    return x.reshape(img.shape).clip(0, 1.0)


def get_augmentations(augmentation_intensity=None, sceptical_augmentation=False, resize=(99,99)):
    crop_limits = (int(resize[0] * 0.85), resize[0])

    if augmentation_intensity == 'light':
        p_augment = 0.25
        p_scale = 0.2
        p_color = 0.25
        p_blur = 0.2
        p_dropout = 0.2
        p_crop = 0.2
        p_flip = 0.2
    elif augmentation_intensity == 'medium':
        p_augment = 0.5
        p_scale = 0.2
        p_color = 0.25
        p_blur = 0.2
        p_dropout = 0.2
        p_crop = 0.2
        p_flip = 0.2
    elif augmentation_intensity == 'heavy':
        p_augment = 0.5
        p_scale = 0.35
        p_color = 0.4
        p_blur = 0.35
        p_dropout = 0.35
        p_crop = 0.35
        p_flip = 0.35

    if augmentation_intensity is None:
        augmentation = None
    elif sceptical_augmentation:
        augmentation = Compose([
            RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=p_color),
            OneOf([Blur(p=1.0), GaussianBlur(p=1.0)], p=p_blur),
            CoarseDropout(max_height=24, max_width=24, p=p_dropout),
            HorizontalFlip(p=p_flip),
            VerticalFlip(p=p_flip),
            RandomSizedCrop(min_max_height=crop_limits,
                            height=resize[0],
                            width=resize[1],
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_CUBIC,
                            p=p_crop)
        ],
            p=p_augment)
    else:
        augmentation = Compose([
            OneOf([
                Rotate(p=1.0, limit=30),
                ShiftScaleRotate(p=1.0, rotate_limit=30),
                # RandomRotate90(p=1.0)
            ],
                p=p_scale),
            RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=p_color),
            OneOf([Blur(p=1.0), GaussianBlur(p=1.0)], p=p_blur),
            CoarseDropout(max_height=24, max_width=24, p=p_dropout),
            HorizontalFlip(p=p_flip),
            VerticalFlip(p=p_flip),
            RandomSizedCrop(min_max_height=crop_limits,
                            height=resize[0],
                            width=resize[1],
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_CUBIC,
                            p=p_crop)
        ],
            p=p_augment)

    return augmentation

import cv2

from albumentations import (
    CLAHE, Blur, CoarseDropout, Compose, GaussianBlur, HorizontalFlip, InvertImg, OneOf, RandomBrightnessContrast,
    RandomRotate90, RandomSizedCrop, Resize, Rotate, ShiftScaleRotate, VerticalFlip)
from seismic.config import config


def get_preprocessing():
    resize = face_config.face_size
    p_resize = 1.0

    preprocess = Compose([Resize(resize, resize, p=p_resize, interpolation=cv2.INTER_CUBIC)])
    return preprocess


def get_augmentations(augmentation_intensity=None, sceptical_augmentation=False):
    resize = 99
    crop_limits = (int(resize * 0.85), resize)

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
            OneOf([
                Rotate(p=1.0, limit=30)
            ], p=p_scale),
            RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=p_color),
            OneOf([Blur(p=1.0), GaussianBlur(p=1.0)], p=p_blur),
            CoarseDropout(max_height=24, max_width=24, p=p_dropout),
            RandomSizedCrop(min_max_height=crop_limits,
                            height=resize,
                            width=resize,
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
                RandomRotate90(p=1.0)
            ],
                p=p_scale),
            RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=p_color),
            OneOf([Blur(p=1.0), GaussianBlur(p=1.0)], p=p_blur),
            CoarseDropout(max_height=24, max_width=24, p=p_dropout),
            HorizontalFlip(p=p_flip),
            VerticalFlip(p=p_flip),
            RandomSizedCrop(min_max_height=crop_limits,
                            height=resize,
                            width=resize,
                            w2h_ratio=1.0,
                            interpolation=cv2.INTER_CUBIC,
                            p=p_crop)
        ],
            p=p_augment)

    return augmentation

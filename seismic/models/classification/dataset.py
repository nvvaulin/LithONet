import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd

import mat4py
from torch.utils.data import Dataset

class FaciesDataset(Dataset):
    """Facies dataset"""

    def __init__(self, images_path, imagesinfo_path, class_name_to_id, preprocess=None, augmentation=None, transform=None,
                 test_mode=False, test_size=40):
        """
        Args:
            images_path     (str) : a path to a folder with images;
            imagesinfo_path (str) : a path to a txt file with info;
            class_name_to_id (dict) : a mapping from string class_name to int class_id
            preprocess      (obj) : albumentations.Compose object;
            augmentation    (obj) : albumentations.Compose object;
            transform       (obj) : torchvision.transforms.Compose object,
                                    required transformations are:
                                        1) transforms.ToTensor(),
                                        2) transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225));
            test_mode       (bool): whether to use only first test_size images from dataset;
            test_size       (int): how many images to use in test_mode;
        """
        self.images_path = images_path
        self.images_info = pd.read_csv(imagesinfo_path)
        self.class_name_to_id = class_name_to_id

        self.labels = list(self.images_info.target)
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.transform = transform
        if test_mode:
            self.images_info = self.images_info.iloc[:test_size]
            self.labels = self.labels[:test_size]

    def __len__(self):
        return self.images_info.shape[0]

    def __getitem__(self, idx):
        filename, label = self.images_info.iloc[idx, [0, 1]]
        try:
            mat = mat4py.loadmat(osp.join(self.images_path, filename))
        except Exception as e:
            print('\n\n\n')
            print(filename)
            raise Exception(e)
        image = np.asarray(mat['img']).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)

        if self.preprocess:
            image = self.preprocess(image=image)['image']
        if self.augmentation:
            image = self.augmentation(image=image)['image']
        if self.transform:
            image = self.transform(image)

        label = self.class_name_to_id[label]
        if type(image) == str:
            print(f'Something went wrong with {filename} image')
        if type(label) == str:
            print(f'Something went wrong with {filename} label')

        return image, label

import cv2
import numpy as np
import os.path as osp
import torch

from albumentations.pytorch import ToTensor

from seismic.data.utils import rle2mask

class TrainDataset():
    def __init__(self, image_dir, train, transform, has_mask=True):
        self.image_dir = image_dir
        self.train = train
        self.transform = transform
        self.has_mask = has_mask
        self.ids = np.unique(train['ImageId'])

    def __getitem__(self, index):
        # img_name = self.train.iloc[index]['ImageId'] ## It takes each image 7 times per epoch
        img_name = self.ids[index]

        path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(path)

        if self.has_mask:
            ce_mask = [
                (i + 1) * rle2mask(rle, shape=(img.shape[0], img.shape[1])) \
                    for i, rle in enumerate(self.train[self.train['ImageId']==img_name]['EncodedPixels'])
            ]
            ce_mask = np.sum(ce_mask, axis=0, dtype=np.float32)

            result = self.transform(
                image=img,
                mask=ce_mask
            )

            result = {
                'image': ToTensor()(image=result['image'])['image'],
                'mask': torch.Tensor(result['mask']),
                'img_name': img_name
            }
            return result

        else:
            init_shape = img.shape
            result = self.transform(
                image=img
            )
            result = {
                'image': ToTensor()(image=result['image'])['image'],
                'img_name': img_name,
                'init_shape': init_shape
            }
            return result

    def __len__(self, ):
        return len(self.ids)

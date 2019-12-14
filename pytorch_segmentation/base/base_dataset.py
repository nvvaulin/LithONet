import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def elastic_transformations(alpha=50, sigma=5, 
                            interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(images):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = images[0].shape
        # Make random fields
        dx = np.random.uniform(-1, 1, image_shape) * alpha
        dy = np.random.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_images = np.array([map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
                                       for image in images])
        return transformed_images
    return _elastic_transform_2D

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False,elastic=False, im_shuffle=False):
        self.root = root
        self.elastic = elastic
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
            self.image_shuffle = im_shuffle
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = np.concatenate([cv2.resize(image[...,i], (w, h), interpolation=cv2.INTER_LINEAR)[...,None] for i in range(image.shape[-1])],axis=-1)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = np.concatenate([cv2.resize(image[...,i], (w, h), interpolation=cv2.INTER_LINEAR)[...,None] for i in range(image.shape[-1])],axis=-1)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = np.concatenate([cv2.warpAffine(image[...,i], rot_matrix, (w, h), flags=cv2.INTER_LINEAR)[:,:,None] for i in range(image.shape[-1])],axis=-1)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = np.concatenate([cv2.copyMakeBorder(image[...,i], value=0, **pad_kwargs)[:,:,None] for i in range(image.shape[-1])],axis=-1)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        #shufle the layers
        if self.image_shuffle:
            a = Layer_shuffle(img=image, mask=label)
            a.get_pair()
            a.shuffle_image()
            image = a.image
            label = a.mask


        if self.elastic:
            transform = elastic_transformations()
            image = np.transpose(transform(np.transpose(image,(2,0,1))),(1,2,0))
            label = transform(label[None,:,:])[0]
        
        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

class Layer_shuffle:
    def __init__(self, img, mask):
        self.image = img
        self.mask = mask
        self.labels = np.unique(mask)
        self.choise = None
        self.artificial_image = None
        self.not_process = 5

    def get_pair(self):
        self.choise = np.random.choice(self.labels, size=(2,))
        while self.not_process in self.choise:
            self.choise = np.random.choice(self.labels, size=(2,))
        self.layer_1 = self.mask == self.choise[0]
        self.layer_2 = self.mask == self.choise[1]

    def shuffle_mask(self):
        """
        change only labels
        :return:
        """
        self.mask[self.layer_1] = self.choise[1]
        self.mask[self.layer_2] = self.choise[0]
        print(self.choise)

    def get_image_region(self, image, bool_mask):
        src1_mask = bool_mask.astype(self.image.dtype)  # change mask to a 3 channel image
        out = cv2.bitwise_and(image, src1_mask)
        out = self.bbox2(out)
        out = self.__fit_borders(out)
        out_mask = self.bbox2(src1_mask)
        out_mask = self.__fit_borders(out_mask)

        return out, out_mask

    def __fit_borders(self, cropped):
        self.im_height, self.im_width = self.image.shape[:2]
        self.arti_height, self.arti_width = cropped.shape[:2]

        if self.arti_height < self.im_height or self.arti_width < self.im_width:
            clac_dif_h = self.im_height - self.arti_height
            clac_dif_w = self.im_width - self.arti_width
            ext_height = (int(clac_dif_h / 2)) * 10 if clac_dif_h > 0 else 0
            top, bottom = ext_height, ext_height
            ext_width = int(clac_dif_w / 2) * 10 if clac_dif_w > 0 else 0
            left, right = ext_width, ext_width
            artificial_image = cv2.copyMakeBorder(cropped, top, bottom, left, right,
                                                  cv2.BORDER_REFLECT_101)
            artificial_image = artificial_image[:self.im_height, :self.im_width]
        if self.arti_height > self.im_height or self.arti_width > self.im_width:
            artificial_image = cropped[:self.im_height, :self.im_width]

        return artificial_image

    def bbox2(self, image):
        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        croped = image[ymin:ymax + 1, xmin:xmax + 1]
        return croped

    def shuffle_image(self):
        """
        change only mask
        :return:
        """
        image_buf = self.image.copy()
        im1, mask_1 = self.get_image_region(image_buf.copy(), self.layer_1.copy())
        print(mask_1.max())
        mask_1[mask_1 > 0] = self.choise[0]
        # print("mask_1.max()", mask_1.max(), self.choise[0])
        im2, mask_2 = self.get_image_region(image_buf.copy(), self.layer_2.copy())
        mask_2[mask_2 > 0] = self.choise[1]
        # something strange here with the mask happens
        # print(self.layer_1.shape, im1.shape, self.layer_2.shape, im2.shape)
        # print(im1[self.layer_1].shape, self.image[self.layer_1].shape)
        self.image[self.layer_1] = im2[self.layer_1]
        self.image[self.layer_2] = im1[self.layer_2]

        self.mask[self.layer_1] = mask_2[self.layer_1]
        self.mask[self.layer_2] = mask_1[self.layer_2]
import numpy as np
import cv2

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
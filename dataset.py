#!/usr/bin/env python3

import os
import logging
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from utils import load_data
from torchvision import transforms
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

np.random.seed(0)

class BaseData(Dataset):

    """Docstring for Dataset. """

    def __init__(self, images, masks, augmentation=False, crop=True, left=145, top=120, crop_size=256):
        self.data_imgs = images
        self.data_masks = masks
        self.scale = 1
        self.augmentation = augmentation
        self.crop = crop
        self.crop_size = crop_size
        self.left = left
        self.top = top

    def __len__(self):
        return len(self.data_imgs)

    def get_annotated_pairs(self):
        """It provides the annotated only data for training
        For now it works only with the train data
        """

        indexes = []
        index = 0
        for image, mask in zip(self.train_imgs, self.train_masks):
            mask = Image.open(mask)
            if np.sum(mask) == 0:
                indexes.append(index)

        # Remove blank images from training
        for i in indexes:
            self.train_imgs.pop(i)
            self.train_masks.pop(i)

    def get_pairs(self, image_paths):
        """
        :returns: TODO

        """
        # TODO it should be generalized for CT as well...
        data_imgs = []
        data_masks = []
        for image_path in image_paths:
            image_id = image_path.split(os.sep)[-3:]
            # Replace image_data file with mask file
            image_index = image_id[-1]
            image_index = "seg_mask_data_" + image_index.lstrip("image_data")
            image_id[-1] = image_index
            mask_path = os.path.join(self.mask_path_MV, (os.sep).join(image_id))
            data_imgs.append(image_path)
            data_masks.append(mask_path)

        return data_imgs, data_masks

    # @classmethod
    def preprocess(self, pil_img, scale):
        w, h = pil_img.size

        if self.crop:
            pil_img = transforms.functional.crop(pil_img, self.top, self.left, self.crop_size, self.crop_size)
        img_nd = np.array(pil_img).astype(float)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            # print(type(img_trans))
            # print("True")
            # print("img max value before scale {}".format(img_trans.max().item()))
            img_trans = img_trans / 255.0
            # print("img max value after scale {}".format(img_trans.max().item()))

        return img_trans

    def __getitem__(self, i):
        img_file = self.data_imgs[i]
        mask_file = self.data_masks[i]
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        assert (
            img.size == mask.size
        ), f"Image and mask {i} should be the same size, but are {img.size} and {mask.size}"

        # Data augmentation
        if self.augmentation:
            if np.random.rand() > 0.5:
                angle = np.random.randint(-10, 10)

                # rotation angle in degree
                # print("img max value before scale {}".format(img.max().item()))
                img = ndimage.rotate(img, angle, reshape=False, axes=(1, 2))
                # print("img max value after scale {}".format(img.max().item()))
                mask = ndimage.rotate(mask, angle, reshape=False, axes=(1, 2))

            if np.random.rand() > 0.5:
                # Numpy to torch.Tensor
                # img = torch.from_numpy(img)
                # mask = torch.from_numpy(mask)
                # img = transforms.functional.hflip(img)
                # mask = transforms.functional.hflip(mask)
                # print("img max value before scale {}".format(img.max().item()))
                img = np.flip(img, axis=2).copy()
                # print("img max value after scale {}".format(img.max().item()))
                mask = np.flip(mask, axis=2).copy()
                # cv2.imwrite('current.png', img[0])
        # plt.imsave('image_1.png', img[0])
        # plt.imsave('mask_1.png', mask[0])


        # Make mask one hot with >1 probs
        # mask = np.expand_dims(mask, axis=1)
        # labels = np.where(mask>0, 1., 0.)
        # mask = np.where(mask==0, 1., 0.)
        # mask = np.concatenate((mask, labels), axis=0)

        # print(mask.shape)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
        # print(mask.shape)
        mask = mask.squeeze(0)

        # print("Image max: {}, Mask max {}".format(img.max().item(), mask.max().item()))
        return {"image": img, "mask": mask, "mask_name": mask_file}

    def create_next_patient(self):
        """Generate the images and masks for the next patient
        """

        keys = [i for i in self.patient_id.keys()]
        key = keys.pop()
        self.image_infer = self.patient_id[key]
        del self.patient_id[key]
        self.image_infer, self.mask_infer = self.get_pairs(self.image_infer)
        for image, mask in zip(self.image_infer, self.mask_infer):
            self.val_imgs.remove(image)
            self.val_masks.remove(mask)

    def augment_set(self, images, masks):
        """Takes as input the images and the masks and adds them to the existing
        dataset lists

        Parameters
        ----------
        images: List with the paths of the images
        masks: List with the paths of the masks

        Returns
        -------
        None
        """
        self.data_imgs.extend(images)
        self.data_masks.extend(masks)


#!/usr/bin/env python3

import os
import logging
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

np.random.seed(0)


class BaseData(Dataset):

    """Docstring for Dataset. """

    def __init__(
        self,
        images,
        masks,
        augmentation=False,
        crop=True,
        left=145,
        top=120,
        crop_size=256,
        n_channels=3,
    ):
        self.data_imgs = images
        self.data_masks = masks
        self.scale = 1
        self.augmentation = augmentation
        self.crop = crop
        self.crop_size = crop_size
        self.left = left
        self.top = top
        self.n_channels = n_channels

        # if self.augmentation:
        #     transforms = [transforms.RandomHorizontalFlip(p=0.5),
        #                   transforms.RandomAffine(degrees=10, translate=(-10, 10), scale=(0.8, 1.2), shear=(-0.1, 0.1))]
        #     self.transforms = transforms.RandomApply(transforms, p=0.5)

    def __len__(self):
        return len(self.data_imgs)

    # @classmethod
    def preprocess(self, pil_img, scale):
        w, h = pil_img.size

        if self.crop:
            pil_img = transforms.functional.crop(
                pil_img, self.top, self.left, self.crop_size, self.crop_size
            )
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

        if self.n_channels > 1:
            img = np.array([])
            for i in img_file:
                channel = np.asarray(Image.open(i))
                channel = np.expand_dims(channel, axis=-1)
                if img.size == 0:
                    img = channel.copy()
                else:
                    img = np.concatenate((img, channel), axis=-1)
        else:
            img = Image.open(img_file)

        img = Image.fromarray(img)
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        # print(type(img), type(mask))
        # assert (
        #     img.size() == mask.size()
        # ), f"Image and mask {i} should be the same size, but are {img.size} and {mask.size}"

        # Data augmentation
        # TODO May not work with more channels (Dimensions)
        if self.augmentation:
            if np.random.rand() > 0.5:
                angle = np.random.randint(-10, 10)
                # img = TF.rotate(img, angle)
                # mask = TF.rotate(mask, angle)
                # affine_transform = TF.affine(degrees=10, translate=(-10, 10), scale=(0.8, 1.2), shear=(-0.1, 0.1))
                # tf
                # rotation angle in degree
                # print("img max value before scale {}".format(img.max().item()))
                img = ndimage.rotate(img, angle, reshape=False, axes=(1, 2))
                # print("img max value after scale {}".format(img.max().item()))
                mask = ndimage.rotate(mask, angle, reshape=False, axes=(1, 2))

            if np.random.rand() > 0.5:
                # Numpy to torch.Tensor
                img = np.flip(img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
                # img = torch.from_numpy(img)
                # mask = torch.from_numpy(mask)
                # img = TF.hflip(img)
                # mask = TF.hflip(mask)

            # if np.random.rand() > 0.5:
            #     scale = np.random.rand(0.8, 1.2)

            # print("img max value before scale {}".format(img.max().item()))
            # img = np.flip(img, axis=2).copy()
            # print("img max value after scale {}".format(img.max().item()))
            # mask = np.flip(mask, axis=2).copy()
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

        print(img_file, mask_file)
        # print("Image max: {}, Mask max {}".format(img.max().item(), mask.max().item()))
        return {"image": img, "mask": mask, "mask_name": mask_file}

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

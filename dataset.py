#!/usr/bin/env python3

import os
import logging
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from utils import load_data

class BaseData(Dataset):

    """Docstring for Dataset. """

    def __init__(self, image_path_CT, image_path_MV, mask_path_CT, mask_path_MV, validation_patients=7, train=True, only_mv=True):
        self.image_path_CT = image_path_CT
        self.mask_path_CT = mask_path_CT
        self.image_path_MV = image_path_MV
        self.mask_path_MV = mask_path_MV
        self.train = train
        self.scale = 1

        # TODO Now its working only with MVCT scans
        if only_mv:
            self.images_mv = load_data(self.image_path_MV)
            self.masks_mv = load_data(self.mask_path_MV)
            self.image_path_MV = (os.sep).join(self.image_path_MV.split(os.sep)[:-3])
            self.mask_path_MV = (os.sep).join(self.mask_path_MV.split(os.sep)[:-3])
        else:
            # TODO add this
            pass

        # TODO Now I have one patient only.... Consider with more patients. I should split along to patients next time
        # Keep the patient's visits here in daily mvct!

        # TODO I havent checked if this works for CT!
        self.patient_id = {}
        # Keep the patient folder codes in self.patient_id
        for image in self.images_mv:
            id_ = image.split(os.sep)[-3]

            if id_ not in self.patient_id.keys():
                key = id_
                self.patient_id[key] = []
            self.patient_id[key].append(image)

        self.train_patients = []
        self.val_patients = []

        print(self.patient_id)
        i = 0
        for key, values in self.patient_id.items():
            if i < validation_patients:
                self.train_patients.extend(values)
            else:
                self.val_patients.extend(values)
            i += 1
        print(self.train_patients)

        # TODO Add also testing phase!!!
        # self.train_patients = self.patient_id[:validation_patients]
        # self.val_patients = self.patient_id[validation_patients:]

        # TODO Fix this!
        # self.images = load_data(self.image_path)
        # self.masks = load_data(self.mask_path)
        if self.train:
            self.data_imgs, self.data_masks = self.get_pairs(self.train_patients)
            # Keep only the images with annotations !!
            print('Train')
            print(len(self.data_imgs), len(self.data_masks))
            self.get_annotated_pairs()
            print(len(self.data_imgs), len(self.data_masks))
            # print(self.data_imgs)
            sys.exit(-1)
        else:
            self.data_imgs, self.data_masks = self.get_pairs(self.val_patients)

        print(len(self.data_imgs), len(self.data_masks))
        assert self.data_imgs != self.data_masks, f'Images and masks should be the same size {len(self.data_imgs)} {len(self.data_masks)}'
        # TODO Shuffle accross days? patients???
        logging.info(f'Creating dataset with {len(self.data_imgs)} examples')

    def __len__(self):
        return len(self.data_imgs)

    def get_annotated_pairs(self):
        """TODO: Docstring for get_annotated_pairs.
        :returns: TODO

        """
        indexes = []
        index = 0
        for image, mask in zip(self.data_imgs, self.data_masks):
            mask = Image.open(mask)
            if np.sum(mask) == 0:
                indexes.append(index)

        # Remove blank images from training
        for i in indexes:
            self.data_imgs.pop(i)
            self.data_masks.pop(i)


    def get_pairs(self, image_paths):
        """TODO: Docstring for get_pairs.
        :returns: TODO

        """
        # TODO it should be generalized for CT as well...
        data_imgs = []
        data_masks = []
        for image_path in image_paths:
            image_id = image_path.split(os.sep)[-3:]
            # Replace image_data file with mask file
            image_index = image_id[-1]
            image_index = 'seg_mask_data_' + image_index.lstrip('image_data')
            image_id[-1] = image_index
            mask_path = os.path.join(self.mask_path_MV, (os.sep).join(image_id))
            data_imgs.append(image_path)
            data_masks.append(mask_path)

        # logging.info('data images {}'.format(data_imgs[0]))
        # logging.info('mask images {}'.format(data_masks[0]))
        return data_imgs, data_masks

        # print(self.images_mv, self.masks_mv)
        # for file, msk_file in zip(self.images_mv, self.masks_mv):
        #     print(file, msk_file)
            # print(file.split(os.sep)[-2].split('_')[0])

            # sys.exit(0)
            # if file.split(os.sep)[-2].split('_')[0] in ids:
            #     data_imgs.append(file)
            #     data_masks.append(msk_file)
        # return data_imgs, data_masks

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.data_imgs[i]
        mask_file = self.data_masks[i]
        # print(i)
        # print(len(img_file), len(mask_file))
        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {i}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {i}: {img_file}'
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

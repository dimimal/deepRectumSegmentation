#!/usr/bin/env python3

import os
import logging
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from utils import load_data

np.random.seed(0)

class BaseData(Dataset):

    """Docstring for Dataset. """

    def __init__(
        self,
        image_path_CT,
        image_path_MV,
        mask_path_CT,
        mask_path_MV,
        train_patients=1,
        validation_patients=7,
        test_patients=1,
        train=True,
        active_learning=False,
        only_mv=True,
    ):
        self.image_path_CT = image_path_CT
        self.mask_path_CT = mask_path_CT
        self.image_path_MV = image_path_MV
        self.mask_path_MV = mask_path_MV
        self.train = train
        self.n_train_patients = train_patients
        self.n_val_patients = validation_patients
        self.n_test_patients = test_patients
        self.scale = 1


        # Keep the paths of images in lists below
        # splitted across patients
        self.train_patients = []
        self.val_patients = []
        self.test_patients = []

        self.train_ids = []
        self.val_ids = []

        # TODO Now its working only with MVCT scans
        if only_mv:
            self.images_mv = load_data(self.image_path_MV)
            self.masks_mv = load_data(self.mask_path_MV)
            self.image_path_MV = (os.sep).join(self.image_path_MV.split(os.sep)[:-3])
            self.mask_path_MV = (os.sep).join(self.mask_path_MV.split(os.sep)[:-3])
        else:
            # TODO add this when I will introduce CT planned as well
            pass

        # TODO Now I have one patient only.... Consider with more patients. I should split along to patients next time
        # Keep the patient's visits here in daily mvct!

        # TODO I havent checked if this works for CT!
        # Add all the patient IDs here! It works as a buffer also!
        self.patient_id = {}
        # Keep the patient folder codes in self.patient_id
        for image in self.images_mv:
            id_ = image.split(os.sep)[-3]
            if id_ not in self.patient_id.keys():
                key = id_
                self.patient_id[key] = []
            self.patient_id[key].append(image)


        # Get the testing patient and drop from patient id
        keys = [i for i in self.patient_id.keys()]
        for i in range(self.n_test_patients):
            key = np.random.choice(keys)
            index = keys.index(key)
            self.test_patients.extend(self.patient_id[key])
            del self.patient_id[key]
            keys.pop(index)

        # Select the test patient ids first!
        i = 0
        for key, values in self.patient_id.items():
            if i < train_patients:
                self.train_patients.extend(values)
                self.train_ids.append(key)
            else:
                self.val_patients.extend(values)
                self.val_ids.append(key)
            i += 1

        # TODO Add also testing phase!!!
        self.train_imgs, self.train_masks = self.get_pairs(self.train_patients)
        self.get_annotated_pairs()
        self.val_imgs, self.val_masks = self.get_pairs(self.val_patients)
        self.test_imgs, self.test_masks = self.get_pairs(self.test_patients)

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
            image_index = "seg_mask_data_" + image_index.lstrip("image_data")
            image_id[-1] = image_index
            mask_path = os.path.join(self.mask_path_MV, (os.sep).join(image_id))
            data_imgs.append(image_path)
            data_masks.append(mask_path)

        return data_imgs, data_masks


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

    def setDataset(self, option='train'):
        """TODO: Docstring for setDataset.
        :returns: TODO

        """
        if option == 'train':
            self.data_imgs = self.train_imgs
            self.data_masks = self.train_masks
        elif option == 'val':
            self.data_imgs = self.val_imgs
            self.data_masks = self.val_masks
        elif option == 'test':
            self.data_imgs = self.test_imgs
            self.data_masks = self.test_masks
        else:
            raise Exception('Uknown parameter {}'.format(option))
        return self

    def __getitem__(self, i):
        img_file = self.data_imgs[i]
        mask_file = self.data_masks[i]
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert (
            img.size == mask.size
        ), f"Image and mask {i} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask)}

    def create_next_patient(self):
        """TODO: Docstring for function.
        """

        keys = [i for i in self.patient_id.keys()]
        key = keys.pop()
        self.image_infer = self.patient_id[key]
        del self.patient_id[key]
        self.image_infer, self.mask_infer = self.get_pairs(self.image_infer)
        for image, mask in zip(self.image_infer, self.mask_infer):
            self.val_imgs.remove(image)
            self.val_masks.remove(mask)

    def infer_patient(self):
        """This is used to infer the next patient obtained from the validation

        """
        self.data_imgs, self.data_masks = self.image_infer,  self.mask_infer

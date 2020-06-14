#!/usr/bin/env python3

"""
File: utils.py
Author: Dimitris Mallios
Email: dimimallios@gmail.com
Description:
"""

import os
import sys
import glob
import re
from scipy import io
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def export_images(image_paths, mask_paths, out_path, keys={'image': 'imageT', 'mask': 'rectumT_seg_man'}, resize=False, sampling_size=(512, 512), extract='ct'):
    """Export images from mat files to png files in png -> image, mask
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_p, mask_p in zip(image_paths, mask_paths):
        print(image_p,mask_p)
        if extract == 'mvct':
            patient_number = image_p.split(os.sep)[-4]
            patient_day = image_p.split(os.sep)[-2]
            print('Processing patient: ', patient_number)
            print('Processing day: ', patient_day)
        elif extract == 'ct':
            patient_number = image_p.split(os.sep)[-3]
            print('Processing patient: ', patient_number)


        # check if patient folder exists!
        if not os.path.exists(os.path.join(out_path, 'images', patient_number)):
            os.makedirs(os.path.join(out_path, 'images', patient_number))
        if not os.path.exists(os.path.join(out_path, 'mask', patient_number)):
            os.makedirs(os.path.join(out_path, 'mask', patient_number))

        # Check if patient's daily scans exist!
        if extract == 'mvct':
            if not os.path.exists(os.path.join(out_path, 'images', patient_number, patient_day)):
                os.makedirs(os.path.join(out_path, 'images', patient_number, patient_day))
            if not os.path.exists(os.path.join(out_path, 'mask', patient_number, patient_day)):
                os.makedirs(os.path.join(out_path, 'mask', patient_number, patient_day))

        mvct_image = io.loadmat(image_p)[keys['image']]
        mvct_convhull = io.loadmat(mask_p)[keys['mask']]
        # print(np.unique(mvct_image))
        # print(np.unique(mvct_convhull))
        # print(mvct_image.min(), mvct_image.max())
        # sys.exit(-1)
        depth_image = mvct_image.shape[-1]

        # print(depth_image)
        # print(mvct_convhull.shape[2])
        for i in range(mvct_convhull.shape[2]):
            # Discard unanntotated images
            # if np.sum(mvct_convhull[:, :, i]) == 0:
            #     continue
            if i > depth_image-1:
                continue
            processed_img = exposure.equalize_adapthist(mvct_image[:, :, i])            # cv2.convertTo(dst, CV_8U, 1.0/256.0)
            # plt.imshow(exposure.equalize_adapthist(mvct_image[:, :, i]))
            # plt.show()
            if extract == 'mvct':
                out_image = os.path.join(out_path, 'images', patient_number, patient_day, 'image_{}_{}.png'.format(patient_number, i))
                out_mask = os.path.join(out_path, 'mask', patient_number, patient_day, 'seg_mask_{}_{}.png'.format(patient_number, i))
            else:
                out_image = os.path.join(out_path, 'images', patient_number, 'image_{}_{}.png'.format(patient_number, i))
                out_mask = os.path.join(out_path, 'mask', patient_number, 'seg_mask_{}_{}.png'.format(patient_number, i))

            if resize:
                resized_mvct = cv2.resize(processed_img, sampling_size, cv2.INTER_CUBIC)
                resized_mvct = np.clip(resized_mvct * 255, 0, 255).astype(np.uint8)
                resized_mvct_mask = cv2.resize(mvct_convhull[:, :, i], sampling_size, cv2.INTER_NEAREST)
                processed_mask = np.where(resized_mvct_mask == 1, 255, 0)
                cv2.imwrite(out_image, resized_mvct)
                cv2.imwrite(out_mask, processed_mask)
            else:
                processed_img = np.clip(processed_img * 255, 0, 255).astype(np.uint8)

                processed_mask = np.where(mvct_convhull[:, :, i] == 1, 255, 0)
                cv2.imwrite(out_image, processed_img)
                cv2.imwrite(out_mask, processed_mask)

def clean_annotations(images, annotations, trim=True):
    """Remove MVct images from the path due to not having
    annotations
    :returns: TODO

    """
    trimmed_images = []
    annot_numbers = []

    # Create the anotation numbers
    if trim:
        for annot in annotations:
            annot_num = re.findall('\d+_\d+.mat', annot)[0].split('.mat')[0]
            annot_numbers.append(annot_num)

        for index, image in enumerate(images):
            match_number = re.findall('\d+_\d+.mat', image)[0].split('.mat')[0]
            if  match_number in annot_numbers:
                trimmed_images.append(image)
    else:
        return images, annotations

    return trimmed_images, annotations

def linear_ramp(x, slope=0.333333):
    """TODO: Docstring for linear_ramp.
    :returns: TODO

    """
    # TODO Fix this!!!!
    return slope*x


def rescale_hu_to_pixels(image):
    """TODO: Docstring for rescae_hu_to_pixels.
    :returns: TODO

    """
    image = image.astype(np.float32)
    image[image<=10] = 0
    image[image<=-130] = 1
    image[image>100] = 0
    image[(image>=30) and (image<=60)] = 1
    image = np.where((image>=10) and (image<30), linear_ramp(image), image)
    return image

def load_data(path):
    """Insert the path and it will return the list with all the files listed
    in the corresponding folder (path). Include the extension
    :returns:
    """

    return sorted(glob.glob(path))

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
from torch.functional import F
from torch import optim
from torch import nn
from tqdm import tqdm
import cv2
import numpy as np
import torch
from skimage import exposure
from PIL import Image
from dice_loss import dice_coeff, dice_loss
import matplotlib.pyplot as plt


def get_annotated_pairs(images, masks):
    """It provides the annotated only data for training
    For now it works only with the train data
    """

    indexes = []
    discard_mask = []
    discard_img = []
    for image, name_mask in zip(images, masks):
        mask = np.array(Image.open(name_mask)).astype(float)
        if np.sum(mask) == 0:
            discard_img.append(image)
            discard_mask.append(name_mask)
        # else:
        #     print('False')

    # Remove blank images from training
    for img, mask  in zip(discard_img, discard_mask):
        try:
            images.remove(img)
            masks.remove(mask)
        except ValueError:
            print(mask)

    return images, masks

def get_pairs(image_paths, dir_masks):
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
            mask_path = os.path.join(dir_masks, (os.sep).join(image_id))
            data_imgs.append(image_path)
            data_masks.append(mask_path)

        return data_imgs, data_masks



def export_images(
    image_paths,
    mask_paths,
    out_path,
    keys={"image": "imageT", "mask": "rectumT_seg_man"},
    resize=False,
    sampling_size=(512, 512),
    extract="ct",
):
    """Export images from mat files to png files in png -> image, mask
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_p, mask_p in zip(image_paths, mask_paths):
        print(image_p, mask_p)
        if extract == "mvct":
            patient_number = image_p.split(os.sep)[-4]
            patient_day = image_p.split(os.sep)[-2]
            print("Processing patient: ", patient_number)
            print("Processing day: ", patient_day)
        elif extract == "ct":
            patient_number = image_p.split(os.sep)[-3]
            print("Processing patient: ", patient_number)

        # check if patient folder exists!
        if not os.path.exists(os.path.join(out_path, "images", patient_number)):
            os.makedirs(os.path.join(out_path, "images", patient_number))
        if not os.path.exists(os.path.join(out_path, "mask", patient_number)):
            os.makedirs(os.path.join(out_path, "mask", patient_number))

        # Check if patient's daily scans exist!
        if extract == "mvct":
            if not os.path.exists(
                os.path.join(out_path, "images", patient_number, patient_day)
            ):
                os.makedirs(
                    os.path.join(out_path, "images", patient_number, patient_day)
                )
            if not os.path.exists(
                os.path.join(out_path, "mask", patient_number, patient_day)
            ):
                os.makedirs(os.path.join(out_path, "mask", patient_number, patient_day))

        mvct_image = io.loadmat(image_p)[keys["image"]]
        mvct_convhull = io.loadmat(mask_p)[keys["mask"]]
        depth_image = mvct_image.shape[-1]

        # print(depth_image)
        # print(mvct_convhull.shape[2])
        for i in range(mvct_convhull.shape[2]):
            # Discard unanntotated images
            # if np.sum(mvct_convhull[:, :, i]) == 0:
            #     continue
            if i > depth_image - 1:
                continue
            processed_img = exposure.equalize_adapthist(
                mvct_image[:, :, i], kernel_size=(24,24), clip_limit=0.005
            )  # cv2.convertTo(dst, CV_8U, 1.0/256.0)
            # processed_img = np.where((processed_img > 20) & (processed_img < 76), 255, processed_img)
            # plt.imshow(exposure.equalize_adapthist(mvct_image[:, :, i]))
            # plt.show()
            if extract == "mvct":
                out_image = os.path.join(
                    out_path,
                    "images",
                    patient_number,
                    patient_day,
                    "image_{}_{}.png".format(patient_number, i),
                )
                out_mask = os.path.join(
                    out_path,
                    "mask",
                    patient_number,
                    patient_day,
                    "seg_mask_{}_{}.png".format(patient_number, i),
                )
            else:
                out_image = os.path.join(
                    out_path,
                    "images",
                    patient_number,
                    "image_{}_{}.png".format(patient_number, i),
                )
                out_mask = os.path.join(
                    out_path,
                    "mask",
                    patient_number,
                    "seg_mask_{}_{}.png".format(patient_number, i),
                )

            if resize:
                resized_mvct = cv2.resize(processed_img, sampling_size, cv2.INTER_CUBIC)
                resized_mvct = np.clip(resized_mvct * 255, 0, 255).astype(np.uint8)
                resized_mvct_mask = cv2.resize(
                    mvct_convhull[:, :, i], sampling_size, cv2.INTER_NEAREST
                )
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
            annot_num = re.findall("\d+_\d+.mat", annot)[0].split(".mat")[0]
            annot_numbers.append(annot_num)

        for index, image in enumerate(images):
            match_number = re.findall("\d+_\d+.mat", image)[0].split(".mat")[0]
            if match_number in annot_numbers:
                trimmed_images.append(image)
    else:
        return images, annotations

    return trimmed_images, annotations


def linear_ramp(x, slope=0.333333):
    """TODO: Docstring for linear_ramp.
    :returns: TODO

    """
    # TODO Fix this!!!!
    return slope * x


def rescale_hu_to_pixels(image):
    """TODO: Docstring for rescae_hu_to_pixels.
    :returns: TODO

    """
    image = image.astype(np.float32)
    image[image <= 10] = 0
    image[image <= -130] = 1
    image[image > 100] = 0
    image[(image >= 30) and (image <= 60)] = 1
    image = np.where((image >= 10) and (image < 30), linear_ramp(image), image)
    return image

def infer_patient(net, loader, device, out_path):
    """This is for the active learning part. Infer the next batch (patient)
    of images and save the predictions to a specdified folder and use them as masks in
    next training.
    """

    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255)
    thickness = 1

    all_pred_masks = []

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, mask_names = batch["image"], batch["mask"], batch["mask_name"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_ids = []
            for path in mask_names:
                mask_id = (os.sep).join(path.split(os.sep)[-3:])
                masks_ids.append(os.path.join(out_path, mask_id))

            with torch.no_grad():
                mask_pred = net(imgs)

            # if net.n_classes > 1:
            #     tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
            pred = F.softmax(mask_pred, dim=1)
            dsc_score = dice_coeff(pred, true_masks).item()
            tot += dsc_score
            pred = torch.argmax(pred, dim=1).float()
            pbar.update()

            imgs = imgs.squeeze(1)
            pred = pred.squeeze(0)
            pred = pred.cpu().numpy()
            imgs = imgs.cpu().numpy()
            # print(imgs.shape)

            # Save the predicted masks
            for index, path in enumerate(masks_ids):
                root_path = (os.sep).join(path.split(os.sep)[:-1])
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                img = np.clip(imgs[index]*255, 0, 255).astype(np.uint8)
                mask = (pred[index]*255).astype(np.uint8)

                # print(img.shape, mask.shape)
                combined = np.concatenate((img, mask), axis=1)
                cv2.putText(combined, str(dsc_score), (0, 180), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.imwrite(path, combined)
                # mask.save(path)
            all_pred_masks.extend(masks_ids)

    net.train()
    return tot / n_val, all_pred_masks

def load_data(path):
    """Insert the path and it will return the list with all the files listed
    in the corresponding folder (path). Include the extension
    :returns:
    """

    return sorted(glob.glob(path))

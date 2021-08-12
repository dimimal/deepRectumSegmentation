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
from src.dice_loss import dice_coeff, dice_loss
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
    for img, mask in zip(discard_img, discard_mask):
        try:
            images.remove(img)
            masks.remove(mask)
        except ValueError:
            print(mask)

    return images, masks


def get_pairs(image_paths, dir_masks):
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

        for i in range(mvct_convhull.shape[2]):
            # Discard unanntotated images
            # if np.sum(mvct_convhull[:, :, i]) == 0:
            #     continue
            if i > depth_image - 1:
                continue
            processed_img = exposure.equalize_adapthist(
                mvct_image[:, :, i], kernel_size=(24, 24), clip_limit=0.005
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


def get_index(name):
    """Returns the index from the image path at the end
    """
    id_index = int(name.split(".")[0].split("_")[-1])
    return id_index


# Order the images
def order_images(images):
    order = []
    ordered_list = []
    prefix = "_".join(images[0].split(".")[0].split("_")[:-1])
    for path in images:
        patient_id = get_index(path)
        order.append(patient_id)
    order = sorted(order)
    for id_ in order:
        ordered_list.append(prefix + "_" + str(id_) + ".png")
    return ordered_list


def get_grouped_pairs(images, masks, n=3):
    """Gets as input a list of images and masks and returns the
        (n_image) - (1-mask [middle one])

        It works only with n=3
    """

    # Save the data in {patient_id: data_name}
    patient_images = {}
    patient_masks = {}
    for img_path, msk_path in zip(images, masks):
        img_index = (os.sep).join(img_path.split(os.sep)[:-1])
        msk_index = (os.sep).join(msk_path.split(os.sep)[:-1])

        img_name = img_path.split(os.sep)[-1]
        mask_name = msk_path.split(os.sep)[-1]

        if img_index not in patient_images.keys():
            patient_images[img_index] = []
            patient_masks[msk_index] = []

        patient_images[img_index].append(img_name)
        patient_masks[msk_index].append(mask_name)

    # Order data
    for key_img, key_msk in zip(patient_images.keys(), patient_masks.keys()):
        images = patient_images[key_img]
        masks = patient_masks[key_msk]
        images = order_images(images)
        masks = order_images(masks)
        patient_images[key_img] = images
        patient_masks[key_msk] = masks

    grouped_imgs = []
    grouped_msks = []
    offset = n // 2
    for key_img, key_msk in zip(patient_images.keys(), patient_masks.keys()):
        images = patient_images[key_img]
        masks = patient_masks[key_msk]

        # Do the mirroring
        first_index = get_index(images[0]) - 1
        last_index = get_index(images[-1]) + 1
        set_image = images[0].split(".")[0].split("_")
        first_image = set_image.copy()
        first_image[-1] = str(first_index)
        set_image = images[-1].split(".")[0].split("_")
        last_image = set_image.copy()
        last_image[-1] = str(last_index)
        first_image = "_".join(first_image) + ".png"
        last_image = "_".join(last_image) + ".png"

        images.insert(0, first_image)
        if os.path.exists(last_image):
            images.append(last_image)
        else:
            images.append(first_image)

        for i in range(len(images)):
            if i == len(masks):
                mid = -1
            else:
                mid = i
            pack = []

            if i + n > len(images):
                break

            for j in range(n):
                try:
                    img = images[i + j]
                except IndexError:
                    print(i + j, len(images))
                image_name = os.path.join(key_img, img)
                pack.append(image_name)

            grouped_imgs.append(pack)

            # Get the middle mask
            mask_name = os.path.join(key_msk, patient_masks[key_msk][mid])
            grouped_msks.append(mask_name)

    assert len(grouped_imgs) == len(grouped_msks)

    with open("pairs.txt", "w") as filehandle:
        for imgs, mask in zip(grouped_imgs, grouped_msks):
            filehandle.write("%s || %s\n" % (imgs, mask))

    return grouped_imgs, grouped_msks


def infer_patient(net, loader, device, out_path, channels=3):
    """This is for the active learning part. Infer the next batch (patient)
    of images and save the predictions to a specdified folder and use them as masks in
    next training.
    """

    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    mid_chan = channels // 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = 255
    thickness = 1

    all_pred_masks = []

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, mask_names = (
                batch["image"],
                batch["mask"],
                batch["mask_name"],
            )
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_ids = []
            for path in mask_names:
                mask_id = (os.sep).join(path.split(os.sep)[-3:])
                masks_ids.append(os.path.join(out_path, mask_id))

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = F.softmax(mask_pred, dim=1)
            dsc_score = dice_coeff(pred, true_masks).item()
            tot += dsc_score
            pred = torch.argmax(pred, dim=1).float()
            pbar.update()

            imgs = imgs.squeeze(1)
            pred = pred.squeeze(0)
            pred = pred.cpu().numpy()
            imgs = imgs.cpu().numpy()

            # Save the predicted masks
            for index, path in enumerate(masks_ids):
                root_path = (os.sep).join(path.split(os.sep)[:-1])
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                if channels > 1:
                    img = np.clip(imgs[index][mid_chan] * 255, 0, 255).astype(np.uint8)
                else:
                    img = np.clip(imgs[index] * 255, 0, 255).astype(np.uint8)
                mask = (pred[index] * 255).astype(np.uint8)

                combined = np.concatenate((img, mask), axis=1)
                cv2.putText(
                    combined,
                    str(dsc_score),
                    (0, 180),
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.imwrite(path, combined)
            all_pred_masks.extend(masks_ids)

    net.train()
    return tot / n_val, all_pred_masks


def vizualize_predictions(image, pred_mask, gt_mask, out_path):
    """With this function we can vizualize the image along with
    the predicted boundary and the ground truth boundary for debugging
    purposes.

    :arg1: TODO
    :returns: TODO

    """
    img_1, pred_contours, hierarchy = cv2.findContours(
        pred_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    img_2, gt_contours, hierarchy = cv2.findContours(
        gt_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image = cv2.drawContours(image, pred_contours, -1, (255, 0, 0), 3)
    image = cv2.drawContours(image, gt_contours, -1, (0, 255, 0), 3)
    cv2.imwrite(out_path, image)


def load_data(path):
    """Insert the path and it will return the list with all the files listed
    in the corresponding folder (path). Include the extension
    :returns:
    """

    return sorted(glob.glob(path))

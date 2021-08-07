#!/usr/bin/env python3

"""
File: infer_model.py
Author: Dimitrios Mallios
Email: dimimallios@gmail.com
Description:

Pass the model and the weights and infer the images along
with the GT mask  boundaries and the predictions
"""

import os
import sys
import cv2
from tqdm import tqdm
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import (
    load_data,
    get_pairs,
    get_annotated_pairs,
    vizualize_predictions,
    get_grouped_pairs,
)
from dataset import BaseData
from unet import UNet

np.random.seed(0)


def infer(net, loader, device, out_path, channels=3):
    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    mid_chan = channels // 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = 255
    thickness = 1

    all_pred_masks = []

    with tqdm(total=n_val, desc="Infer", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, mask_names = (
                batch["image"],
                batch["mask"],
                batch["mask_name"],
            )
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # TODO Refactor this, make it a function
            masks_ids = []
            for path in mask_names:
                mask_id = (os.sep).join(path.split(os.sep)[-3:])
                masks_ids.append(os.path.join(out_path, mask_id))

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.argmax(mask_pred, dim=1).float()
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
                img = np.clip(imgs[index] * 255, 0, 255).astype(np.uint8)
                if channels > 1:
                    img = np.clip(imgs[index][mid_chan] * 255, 0, 255).astype(np.uint8)
                else:
                    img = np.clip(imgs[index] * 255, 0, 255).astype(np.uint8)
                mask = (pred[index] * 255).astype(np.uint8)

                # mask = (pred[index]*255).astype(np.uint8)
                gt_mask = true_masks[index].cpu().numpy().astype(np.uint8)
                vizualize_predictions(img, mask, gt_mask, path)


def main(args):
    log_file = args[1]
    weights = args[2]
    input_images = args[3]
    mask_path = args[4]
    out_path = args[5]

    # Get pair paths
    image_paths = load_data(input_images)
    mask_paths = load_data(mask_path)

    assert len(image_paths) == len(mask_paths)

    # print(image_paths)
    # images, masks = get_pairs(image_paths, mask_paths)

    images, masks = get_annotated_pairs(image_paths, mask_paths)
    images, masks = get_grouped_pairs(image_paths, mask_paths)

    dataset = BaseData(images, masks)
    data_loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
    )

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net.load_state_dict(torch.load(weights, map_location=device))
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
    )

    infer(net, data_loader, device, out_path)


if __name__ == "__main__":
    main(sys.argv)

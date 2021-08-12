#!/usr/bin/env python3

import os
import logging
import torch
import math
import sys
import numpy as np
import json
from collections import OrderedDict
from models.unet import UNet
from torch.functional import F
from torch import optim
import matplotlib.pyplot as plt
from torch import nn
from kornia.losses import TverskyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import BaseData
from src.dice_loss import dice_coeff, dice_loss, iou_metric
from src.utils import (
    get_annotated_pairs,
    get_pairs,
    load_data,
    infer_patient,
    get_grouped_pairs,
)

np.random.seed(0)

def train_network(
    net, device, cfg, save_cp=True, img_scale=1.0,
):

    N_test_patients = cfg["N_test"]
    N_val_patients = cfg["N_val"]
    N_train_patients = cfg["N_train"]
    isAugment = bool(int(cfg["augmentation"]))
    addSynthetic = bool(int(cfg["synthetic"]))

    dir_images_MV = cfg["dir_images_mv"]
    dir_masks_MV = cfg["dir_masks_mv"]
    output_dir = cfg["output_dir"]
    dir_out_masks = os.path.join(output_dir, cfg["dir_out_masks"])

    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    lr_pen = float(cfg["lr_pen"])
    reg = float(cfg["reg"])

    # Put all the train val test scores here for each epoch
    scores = {}
    scores["train_dsc"] = []
    scores["train_iou"] = []
    scores["val_dsc"] = []
    scores["test_dsc"] = []
    scores["train_loss"] = []
    scores["val_loss"] = []
    scores["test_loss"] = []
    scores["test"] = []

    # Get pair paths
    image_paths = load_data(dir_images_MV)
    # mask_paths = load_data(dir_masks_MV)

    # TODO Do I need that???
    # dir_images_MV = (os.sep).join(dir_images_MV.split(os.sep)[:-3])
    trimmed_masks_MV = (os.sep).join(dir_masks_MV.split(os.sep)[:-3])

    # TODO The code blocks below should be functioned!
    # Split train val test
    # Add all the patient IDs here! It works as a buffer also!
    patient_id = OrderedDict()

    # Keep the patient folder codes in patient_id
    for image in image_paths:
        id_ = image.split(os.sep)[-3]
        if id_ not in patient_id.keys():
            key = id_
            patient_id[key] = []
        patient_id[key].append(image)

    # Get the testing patient and drop from patient id
    test_patients = []
    keys = [i for i in patient_id.keys()]
    for i in range(N_test_patients):
        key = np.random.choice(keys)
        test_patients.extend(patient_id[key])
        del patient_id[key]
        keys.remove(key)

    train_patients = []
    val_patients = []
    train_ids = []
    val_ids = []

    # Select the test patient ids first!
    i = 0
    # for key, values in patient_id.items():
    for i in range(N_train_patients):
        key = np.random.choice(keys)
        train_patients.extend(patient_id[key])
        train_ids.append(key)
        keys.remove(key)
        del patient_id[key]
        i += 1

    i = 0
    # for key, values in patient_id.items():
    for i in range(N_val_patients):
        key = np.random.choice(keys)
        val_patients.extend(patient_id[key])
        val_ids.append(key)
        del patient_id[key]
        keys.remove(key)

    train_images, train_masks = get_pairs(train_patients, trimmed_masks_MV)
    train_images, train_masks = get_annotated_pairs(train_images, train_masks)
    val_images, val_masks = get_pairs(val_patients, trimmed_masks_MV)
    val_images, val_masks = get_annotated_pairs(val_images, val_masks)
    test_images, test_masks = get_pairs(test_patients, trimmed_masks_MV)
    test_images, test_masks = get_annotated_pairs(test_images, test_masks)

    if net.n_channels > 1:
        train_images, train_masks = get_grouped_pairs(train_images, train_masks, n=3)
        val_images, val_masks = get_grouped_pairs(val_images, val_masks, n=3)
        test_images, test_masks = get_grouped_pairs(test_images, test_masks, n=3)

    # val_images, val_masks = get_pairs(val_patients, trimmed_masks_MV)
    # test_images, test_masks = get_pairs(test_patients, trimmed_masks_MV)
    # test_images, test_masks = get_annotated_pairs(test_images, test_masks)

    # TODO Add a bias to the testing patient, This is for testing only
    # bias_images = test_images[:150]
    # bias_masks = test_masks[:150]
    # test_images = test_images[150:]
    # test_masks = test_masks[150:]

    train_dataset = BaseData(
        train_images, train_masks, augmentation=isAugment, n_channels=net.n_channels
    )
    validation_dataset = BaseData(
        val_images, val_masks, augmentation=False, n_channels=net.n_channels
    )
    test_dataset = BaseData(
        test_images, test_masks, augmentation=False, n_channels=net.n_channels
    )
    # n_train = len(train_dataset)

    # Add synthetic images during training
    if addSynthetic:
        synth_images = sorted(load_data(cfg["synth_imgs"]))
        synth_masks = sorted(load_data(cfg["synth_masks"]))
        train_dataset.augment_set(synth_images, synth_masks)
        print(synth_images[:2], synth_masks[:2])
        logging.info("Adding synthetic data in training")

    n_train = len(train_dataset)
    n_val = len(validation_dataset)
    n_test = len(test_dataset)
    print(n_train)
    print(n_val)
    print(n_test)
    sys.exit(-1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # validation_dataset = dataset.setDataset(option="val")
    # test_dataset = dataset.setDataset(option="test")
    # validation_dataset = BaseData(
    #     dir_images_CT, dir_images_MV, dir_masks_CT, dir_masks_MV, train=False
    # )
    n_train = len(train_dataset)
    n_val = len(validation_dataset)
    print(n_train)
    print(n_val)
    print(n_test)

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0
    best_val_score = 0
    early_stop = 9
    no_improve_counter = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=lr_pen)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=2, verbose=True
    )

    # weights = torch.Tensor([0., 10.]).to(device)

    # criterion_1 = nn.BCEWithLogitsLoss(pos_weight=weights)
    if cfg["loss"] == "dsc":
        criterion = dice_loss
    elif cfg["loss"] == "tversky":
        criterion = TverskyLoss(alpha=0.7, beta=0.3)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        total_train_dice_coeff = 0
        total_train_iou = 0
        count = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                count += 1
                imgs = batch["image"]
                true_masks = batch["mask"]
                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                train_dsc = dice_coeff(masks_pred, true_masks).item()
                train_iou = iou_metric(masks_pred, true_masks).item()
                total_train_dice_coeff += train_dsc
                total_train_iou += train_iou

                writer.add_scalar("Loss/train", loss.item(), global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # Add clipping
                if reg > 0:
                    nn.utils.clip_grad_value_(net.parameters(), reg)
                optimizer.step()
                pbar.update(imgs.shape[0])

        val_score, val_loss = eval_net(net, val_loader, device, criterion)
        scheduler.step(val_score)

        print(f"val score: {val_score}")
        logging.info(f"Validation dice score: {val_score}")
        print("Train Dice {}".format(total_train_dice_coeff / count))
        scores["train_dsc"].append(total_train_dice_coeff / count)
        scores["train_iou"].append(total_train_iou / count)
        scores["val_dsc"].append(val_score)
        scores["train_loss"].append(epoch_loss)
        scores["val_loss"].append(val_loss)

        logging.info(f"Iou train score {total_train_iou / count}")
        test_score, _ = infer_patient(
            net, test_loader, device, dir_out_masks, channels=net.n_channels
        )
        print(f"infer test_score {test_score}")

        # Infer the test set
        scores["test"].append(test_score)
        logging.info(f"Test Score {test_score}")

        # train_dsc, _ = infer_patient(
        #     net, train_loader, device, "pred_train_masks_256", channels=net.n_channels
        # )
        logging.info(f"Train DSC accuracy {train_dsc}")

        if val_score > best_val_score:
            best_val_score = val_score
            no_improve_counter = 0
            try:
                os.mkdir(output_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(
                net.state_dict(),
                os.path.join(output_dir, f"CP_{net.n_channels}_bestmodel.pth"),
            )
            logging.info(f"Best Checkpoint {epoch + 1} saved !")
        else:
            no_improve_counter += 1

            if no_improve_counter == early_stop:
                logging.info(
                    f"No validation improvement in epoch {epoch}. Train stoped!"
                )
                break

    writer.close()

    with open(os.path.join(output_dir, cfg["result_file"]), "w") as fp:
        json.dump(scores, fp)


def eval_net(net, loader, device, loss):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    epoch_loss = 0
    criterion = loss
    count = 0
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
            count += 1
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                loss = criterion(mask_pred, true_masks)
                epoch_loss += loss.item()

            dsc = dice_coeff(mask_pred, true_masks).item()
            if math.isnan(dsc):
                dsc = 0.0
            tot += dsc
            pbar.update()

    net.train()
    return tot / count, epoch_loss


def main(args):

    with open(args[1], "r") as f:
        cfg = json.load(f)
    log_file = os.path.join(cfg["output_dir"], cfg["log_file"])

    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    net = UNet(n_channels=cfg["n_channels"], n_classes=cfg["n_classes"], bilinear=True)
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
    )

    # f'\t{'Bilinear' if {net.bilinear} else 'Transposed conv'} upscaling')

    if int(cfg["load"]):
        if os.path.exists(cfg["weights"]):
            net.load_state_dict(torch.load(cfg["weights"], map_location=device))
            logging.info("Model loaded from {}".format(cfg["weights"]))

    net.to(device=device)

    try:
        train_network(net=net, device=device, cfg=cfg, save_cp=True, img_scale=1.0)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main(sys.argv)

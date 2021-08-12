#!/usr/bin/env python3

import os
import logging
import torch
import sys
import json
from PIL import Image
import numpy as np
import math
from models.unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch import optim
from torch import nn
from tqdm import tqdm
from kornia.losses import TverskyLoss
from torch.utils.data import DataLoader
from src.dataset import BaseData
from src.dice_loss import dice_coeff, DiceCoeff, dice_loss, iou_metric
from src.utils import load_data, get_annotated_pairs, get_pairs, infer_patient, get_grouped_pairs


np.random.seed(0)

# TODO Add synthetic data option as well!!

def get_scores_dict():
    """Returns the empty scores to init each cycle score dict
    """
    scores = {}
    scores["train_dsc"] = []
    scores["train_iou"] = []
    scores["val_dsc"] = []
    scores["test_dsc"] = []
    scores["train_loss"] = []
    scores["val_loss"] = []
    scores["test_loss"] = []
    # scores["test"] = []
    return scores

def train_network(
    net, device, cfg=None, save_cp=True, img_scale=1.0,
):

    if cfg is None:
        raise Exception("Config file not given.")

    N_test_patients = cfg["N_test"]
    N_val_patients = cfg["N_val"]
    N_train_patients = cfg["N_train"]
    isAugment = bool(int(cfg["augmentation"]))
    addSynthetic = bool(int(cfg["synthetic"]))
    n_channels = int(cfg['n_channels'])

    dir_images_MV = cfg["dir_images_mv"]
    dir_masks_MV = cfg["dir_masks_mv"]
    output_dir = cfg["output_dir"]
    dir_out_masks = os.path.join(output_dir, cfg["dir_out_masks"])

    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    lr_pen = float(cfg["lr_pen"])
    reg = float(cfg["reg"])

    # Place the scores here from each cycle
    scores = {}

    # Get pair paths
    image_paths = load_data(dir_images_MV)
    mask_paths = load_data(dir_masks_MV)

    # dir_images_MV = (os.sep).join(dir_images_MV.split(os.sep)[:-3])
    trimmed_masks_MV = (os.sep).join(dir_masks_MV.split(os.sep)[:-3])

    # TODO The code blocks below should be functioned!
    # Split train val test
    # Add all the patient IDs here! It works as a buffer also!
    patient_id = {}
    # Keep the patient folder codes in patient_id
    for image in image_paths:
        id_ = image.split(os.sep)[-3]
        if id_ not in patient_id.keys():
            key = id_
            patient_id[key] = []
        patient_id[key].append(image)

    # Get the testing patient and drop from patient id
    # Select the test patient ids first!
    test_patients = []
    keys = [i for i in patient_id.keys()]
    for i in range(N_test_patients):
        key = np.random.choice(keys)
        print(f'Test patient Added: {key}')
        test_patients.extend(patient_id[key])
        del patient_id[key]
        keys.remove(key)

    train_patients = []
    val_patients = []
    train_ids = []
    val_ids = []

    i = 0
    for i in range(N_val_patients):
        key = np.random.choice(keys)
        val_patients.extend(patient_id[key])
        val_ids.append(key)
        print(f'Val patient Added: {key}')
        del patient_id[key]
        keys.remove(key)

    i = 0
    for i in range(N_train_patients):
        key = np.random.choice(keys)
        train_patients.extend(patient_id[key])
        train_ids.append(key)
        keys.remove(key)
        print(f'Train patient Added: {key}')
        del patient_id[key]
        i += 1
    # print(test_patients)

    train_images, train_masks = get_pairs(train_patients, trimmed_masks_MV)
    train_images, train_masks = get_annotated_pairs(train_images, train_masks)
    val_images, val_masks = get_pairs(val_patients, trimmed_masks_MV)
    test_images, test_masks = get_pairs(test_patients, trimmed_masks_MV)

    if n_channels > 1:
        train_images, train_masks = get_grouped_pairs(train_images, train_masks, n=3)
        val_images, val_masks = get_grouped_pairs(val_images, val_masks, n=3)
        test_images, test_masks = get_grouped_pairs(test_images, test_masks, n=3)

    train_dataset = BaseData(train_images, train_masks, n_channels=n_channels, augmentation=isAugment)
    validation_dataset = BaseData(val_images, val_masks, n_channels=n_channels)
    test_dataset = BaseData(test_images, test_masks, n_channels=n_channels)

    # This changes during Learning!
    n_train = len(train_dataset)
    n_val = len(validation_dataset)
    n_test = len(test_dataset)

    # Init the dataloaders!
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}
    # """
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=lr_pen)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )

    if cfg["loss"] == "dsc":
        criterion = dice_loss
    elif cfg["loss"] == "tversky":
        criterion = TverskyLoss(alpha=0.7, beta=0.3)

    round_ = 0
    # test_score, out_masks = infer_patient(net, val_loader, device, dir_out_masks, mode='active', channels=n_channels)

    # This is the cycle
    while bool(patient_id):
        print("Start Trraining... \n")
        print("Train size ", n_train)
        print("Round ", round_)

        temp_dict = get_scores_dict()
        scores[round_] = temp_dict

        train_acc = train_model(
                        net,
                        epochs,
                        device,
                        n_train,
                        train_loader,
                        val_loader,
                        train_dataset,
                        criterion,
                        scheduler,
                        optimizer,
                        writer,
                        global_step,
                        batch_size,
                        reg,
                        lr_pen,
                        save_cp=True
                    )
        #
        scores[round_]['train_dsc'].append(train_acc)

        print("Init testing...")
        test_score = eval_net(net, test_loader, device, criterion)
        scores[round_]['test_dsc'].append(test_score)

        print("Test score : {}".format(test_score))

        print("Train Finish, begin infer on the next batch and retrain!")
        # Create the new loader for inference
        key = np.random.choice(keys)
        print("Iferring Patient id {}".format(key))
        keys.remove(key)
        next_batch = patient_id[key]
        del patient_id[key]
        images, masks = get_pairs(next_batch, trimmed_masks_MV)
        infer_dataset = BaseData(images, masks, n_channels=n_channels)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        test_score, out_masks = infer_patient(net, infer_loader, device, dir_out_masks, mode='active', channels=n_channels)
        print("Score in the inferring patient {} is {}".format(key, test_score))

        # new_images, new_masks = get_pairs(images, out_masks)
        new_images, new_masks = get_annotated_pairs(images, out_masks)
        print("New annotated masks: ", len(new_masks))

        # Add the new images on the train dataset
        train_dataset.augment_set(new_images, new_masks)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        n_train = len(train_dataset)
        # Next train less epochs!
        epochs = 2
        round_ += 1

    print("That was the last inferred batch!")

    test_score = eval_net(net, test_loader, device, criterion)
    print("Final Test score : {}".format(test_score))

    #
    with open(os.path.join(output_dir, cfg["result_file"]), "w") as fp:
        json.dump(scores, fp)


def train_model(
    net,
    epochs,
    device,
    n_train,
    train_loader,
    val_loader,
    train_dataset,
    criterion,
    scheduler,
    optimizer,
    writer,
    global_step,
    batch_size,
    reg,
    lr_pen,
    save_cp=True,
):

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        total_dsc = 0
        counter = 0
        all_total_dsc = []
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
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
                dsc = dice_coeff(masks_pred, true_masks)

                total_dsc += dsc
                #
                epoch_loss += loss.item()
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping
                if reg > 0:
                    nn.utils.clip_grad_value_(net.parameters(), reg)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
            total_dsc /= counter
            all_total_dsc.append(total_dsc)

        # Evaluate val score
        val_score, _ = eval_net(net, val_loader, device, criterion)
        scheduler.step(val_score)

        # TODO USE BEST eval score in the future!
        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info(f"Created checkpoint directory")
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(), os.path.join(dir_checkpoint, f"CP_epoch{epoch + 1}.pth"))
        #     logging.info(f"Checkpoint {epoch + 1} saved !")

    return all_total_dsc


def eval_net(net, loader, device, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    epoch_loss = 0
    counter = 0
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
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
            counter += 1

    net.train()
    return tot / counter, epoch_loss / counter


def main(args):
    """TODO: Docstring for main.
    :returns: TODO

    """

    with open(args[1], "r") as f:
        cfg = json.load(f)
    log_file = os.path.join(cfg["output_dir"], cfg["log_file"])

    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode="w",
        format="%(levelname)s: %(message)s",
    )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = cfg['device']
    logging.info(f"Using device {device}")
    # device = 'cpu'

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(
        n_channels=cfg["n_channels"], n_classes=int(cfg["n_classes"]), bilinear=True
    )
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
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_network(net=net, device=device, img_scale=1, cfg=cfg)  # args.scale,

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main(sys.argv)

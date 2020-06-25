#!/usr/bin/env python3

import os
import logging
import torch
import sys
import numpy as np
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import BaseData
import segmentation_models_pytorch as smp
from dice_loss import dice_coeff


dir_images_CT = (
    "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/images/**/*.png"
)
dir_images_MV = (
    "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/images/**/**/*.png"
)
dir_masks_CT = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/mask/**/*.png"
dir_masks_MV = (
    "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/mask/**/**/*.png"
)

dir_checkpoint = "./Checkpoints"


def train_network(
    net,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    val_percent=0.1,
    save_cp=True,
    img_scale=1.0,
):

    dataset = BaseData(
        dir_images_CT,
        dir_images_MV,
        dir_masks_CT,
        dir_masks_MV,
        train_patients=1,
        validation_patients=8,
        test_patients=1,
        train=True,
        active_learning=False,
    )

    # TODO This should be change during Learning!
    n_train = len(dataset.setDataset(option="train"))
    n_val = len(dataset.setDataset(option="val"))
    n_test = len(dataset.setDataset(option="test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    n_train = len(train_dataset)
    print(n_train)
    validation_dataset = dataset.setDataset(option="val")
    test_dataset = dataset.setDataset(option="test")
    # validation_dataset = BaseData(
    #     dir_images_CT, dir_images_MV, dir_masks_CT, dir_masks_MV, train=False
    # )
    # n_val = int(len(dataset) * val_percent)
    n_train = len(train_dataset)
    n_val = len(validation_dataset)
    print(n_train)
    print(n_val)
    # train, val = random_split(dataset, [n_train, n_val])
    print(train_loader.__len__())
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {lr}
    #     Training size:   {n_train}
    #     Validation size: {n_val}
    #     Checkpoints:     {save_cp}
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    # ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if net.n_classes > 1 else "max", patience=2
    )
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
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
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(train_dataset) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), global_step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                        )
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        "learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )

                    if net.n_classes > 1:
                        logging.info("Validation cross entropy: {}".format(val_score))
                        writer.add_scalar("Loss/test", val_score, global_step)
                    else:
                        logging.info("Validation Dice Coeff: {}".format(val_score))
                        writer.add_scalar("Dice/test", val_score, global_step)

                    writer.add_images("images", imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images("masks/true", true_masks, global_step)
                        writer.add_images(
                            "masks/pred", torch.sigmoid(masks_pred) > 0.5, global_step
                        )

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


def infer_patient(net, loader, device, out_path):
    """This is for the active learning part. Infer the next batch (patient)
    of images and save the predictions to a specdified folder and use them as masks in
    next training.

    """
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                # TODO Here I have the mask! I need the path though too!
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

    pass


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
    )
    # f'\t{'Bilinear' if {net.bilinear} else 'Transposed conv'} upscaling')

    # if args.load:
    #     net.load_state_dict(
    #         torch.load(args.load, map_location=device)
    #     )
    #     logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_network(
            net=net,
            epochs=20,  # args.epochs,
            batch_size=32,  # args.batchsize,
            lr=0.0001,  # args.lr,
            device=device,
            img_scale=1,  # args.scale,
            val_percent=1,
        )  # args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()

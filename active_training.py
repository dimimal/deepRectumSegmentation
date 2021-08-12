#!/usr/bin/env python3

"""
This script is used to train the U-Net model using the online Training mode
"""


import os
import logging
import torch
import sys
from PIL import Image
import numpy as np
from models.unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import BaseData
from src.dice_loss import dice_coeff, DiceCoeff
from src.utils import load_data, get_annotated_pairs, get_pairs

# These are the paths being used for training
dir_images_CT = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/images/**/*.png"
dir_images_MV = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/images/**/**/*.png"
dir_masks_CT = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/mask/**/*.png"
dir_masks_MV = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/mask/**/**/*.png"
dir_out_masks = "/home/dimitris/SOTON_COURSES/Msc_Thesis/predicted_masks"


dir_checkpoint = "./Checkpoints"
N_test_patients = 1
N_train_patients = 3
N_val_patients = 1


def train_network(
    net,
    device,
    epochs=2,
    batch_size=1,
    lr=0.001,
    val_percent=0.1,
    save_cp=True,
    img_scale=1.0,
):

    # Get pair paths
    image_paths = load_data(dir_images_MV)
    mask_paths = load_data(dir_masks_MV)
    trimmed_masks_MV = (os.sep).join(dir_masks_MV.split(os.sep)[:-3])

    patient_id = {}
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
    test_images, test_masks = get_pairs(test_patients, trimmed_masks_MV)
    train_dataset = BaseData(train_images, train_masks)
    validation_dataset = BaseData(val_images, val_masks)
    test_dataset = BaseData(test_images, test_masks)

    #
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
        Device:          {device.type}
        Images scaling:  {img_scale}
     """
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    round_ = 0
    # This is the patient cycle
    while bool(patient_id):
        print("Start Trraining... \n")
        print("Train size ", n_train)
        print("Round ", round_)
        train_model(
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
            save_cp=True,
        )
        print("Init testing...")
        test_score = eval_net(net, test_loader, device)
        print("Test score : {}".format(test_score))

        print("Train Finish, begin infer on the next batch and retrain!")
        # Create the new loader for inference
        key = np.random.choice(keys)
        print("Iferring Patient id {}".format(key))
        keys.remove(key)
        next_batch = patient_id[key]
        del patient_id[key]
        images, masks = get_pairs(next_batch, trimmed_masks_MV)
        infer_dataset = BaseData(images, masks)
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        test_score, out_masks = infer_patient(net, infer_loader, device, dir_out_masks)
        print("Score in the inferring patient {} is {}".format(key, test_score))

        # print(np.sum([mask.cpu().detach().numpy() for mask in out_masks ]))
        # new_images, new_masks = get_pairs(images, out_masks)
        new_images, new_masks = get_annotated_pairs(images, out_masks)

        print("New annotated masks: ", len(new_masks))
        # add the new images on the train dataset
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
        epochs = 4
        round_ += 1
    print("That was the last inferred batch!")

    test_score = eval_net(net, test_loader, device)
    print("Final Test score : {}".format(test_score))


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
    save_cp=True,
):
    """
    :args:
        net: The network model torch.nn.model
        epochs: The number of epochs
        device: Either cpu or gpu
        n_train: The amount of train samples
        train_loader: The train torch loader (dataset.Dataloader)
        val_loader: The val torch loader (dataset.Dataloader)
        train_dataset: The torch dataset
        criterion: The loss function
        scheduler: The scheduler to use during training
        writer: The logger object
        global_step: The

    :returns: None

    """
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
                # print(loss.item(), loss.grad)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (len(train_dataset) // 1) == 0:
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


@torch.no_grad()
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient

    """

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

    try:
        train_network(
            net=net,
            epochs=40,  # args.epochs,
            batch_size=32,  # args.batchsize,
            lr=0.001,
            device=device,
            img_scale=1,  # args.scale,
            val_percent=1,
        )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import logging
import torch
import math
import sys
import numpy as np
import json
from collections import OrderedDict
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F
from torch import optim
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import BaseData
import segmentation_models_pytorch as smp
from dice_loss import dice_coeff, dice_loss, iou_metric
from utils import get_annotated_pairs, get_pairs, load_data, infer_patient

np.random.seed(0)

# dir_images_CT = (
#     "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/images/**/*.png"
# )
# dir_images_MV = (
#     "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/images/**/**/*.png"
# )
# dir_masks_CT = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_Plan/mask/**/*.png"
# dir_masks_MV = (
#     "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_Del/mask/**/**/*.png"
# )
dir_images_MV = "/home/dimitris/SOTON/MSc_Project/data/MVCT_1/images/**/**/*.png"
dir_masks_MV = "/home/dimitris/SOTON/MSc_Project/data/MVCT_1/mask/**/**/*.png"

dir_images_CT = "/home/dimitris/SOTON/MSc_Project/data/CT_Plan/images/**/*.png"
# dir_images_MV = "/home/dimitris/SOTON/MSc_Project/data/MVCT_Del/images/**/**/*.png"
dir_masks_CT = "/home/dimitris/SOTON/MSc_Project/data/CT_Plan/mask/**/*.png"
# dir_masks_MV = "/home/dimitris/SOTON/MSc_Project/data/MVCT_Del/mask/**/**/*.png"
dir_out_masks = "/home/dimitris/SOTON/MSc_Project/predicted_masks_256"



N_test_patients = 1
N_val_patients = 1
N_train_patients = 3

dir_checkpoint = "./Checkpoints"


def train_network(
    net,
    device,
    epochs=10,
    batch_size=32,
    lr=0.0001,
    val_percent=0.1,
    save_cp=True,
    img_scale=1.0,
):

    # Put all the train val test scores here for each epoch
    scores = {}
    scores['train'] = []
    scores['val'] = []
    scores['test'] = []

    losses = {}
    losses['train'] = []
    losses['val'] = []
    losses['test'] = []

    # Get pair paths
    image_paths = load_data(dir_images_MV)
    mask_paths = load_data(dir_masks_MV)

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
    test_images, test_masks = get_pairs(test_patients, trimmed_masks_MV)
    test_images, test_masks = get_annotated_pairs(test_images, test_masks)

    # TODO Add a bias to the testing patient, This is for testing only
    # bias_images = test_images[:150]
    # bias_masks = test_masks[:150]
    # test_images = test_images[150:]
    # test_masks = test_masks[150:]

    print(len(test_images))
    train_dataset = BaseData(train_images, train_masks)
    validation_dataset = BaseData(val_images, val_masks)
    test_dataset = BaseData(test_images, test_masks)
    print(len(test_images))
    # sys.exit(-1)

    # TODO Adding test bias
    # train_dataset.augment_set(bias_images, bias_masks)

    # TODO This should be change during Learning!
    n_train = len(train_dataset)
    n_val = len(validation_dataset)
    n_test = len(test_dataset)

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
    # n_val = int(len(dataset) * val_percent)
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

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=2)

    weights = torch.Tensor([0., 10.]).to(device)
    if net.n_classes > 1:
        criterion_1 = nn.CrossEntropyLoss(weight=weights)
        # criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = dice_loss
    else:
        criterion_1 = nn.BCEWithLogitsLoss(pos_weight=weights)
        criterion_2 = dice_loss
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
                loss_2 = criterion_2(masks_pred, true_masks)
                loss = loss_2
                epoch_loss += loss.item()

                train_dsc = dice_coeff(masks_pred, true_masks).item()
                train_iou = iou_metric(masks_pred, true_masks).item()
                # print(train_dsc)
                total_train_dice_coeff += train_dsc
                total_train_iou += train_iou

                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])

                # TODO This is for the tensorboard
                # global_step += 1
                # if global_step % (len(train_dataset) // (2 * batch_size)) == 0:
                #     for tag, value in net.named_parameters():
                #         tag = tag.replace(".", "/")
                #         writer.add_histogram(
                #             "weights/" + tag, value.data.cpu().numpy(), global_step
                #         )
                #         writer.add_histogram(
                #             "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                #         )
                #     val_score, val_loss = eval_net(net, val_loader, device)
                #     scheduler.step(val_score)
                #     print(f"val score: {val_score}")
                #     logging.info(f"Validation dice score: {val_score}")
                #     writer.add_scalar(
                #         "learning_rate", optimizer.param_groups[0]["lr"], global_step
                #     )

                #     if net.n_classes > 1:
                #         logging.info("Validation Loss: {}".format(val_loss))
                #         writer.add_scalar("Loss/test", val_score, global_step)
                #     else:
                #         logging.info("Validation Dice Coeff: {}".format(val_score))
                #         writer.add_scalar("Dice/test", val_score, global_step)

                #     writer.add_images("images", imgs, global_step)
                #     if net.n_classes == 1:
                #         writer.add_images("masks/true", true_masks, global_step)
                #         writer.add_images(
                #             "masks/pred", torch.sigmoid(masks_pred) > 0.5, global_step
                #         )


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

        val_score, val_loss = eval_net(net, val_loader, device)
        scheduler.step(val_score)
        print(f"val score: {val_score}")
        logging.info(f"Validation dice score: {val_score}")
        # print('Dice {}'.format(total_train_dice_coeff / n_train))
        scores['train'].append(total_train_dice_coeff / count)
        scores['train_iou'].append(total_train_iou / count)
        scores['val'].append(val_score)
        losses['train'].append(epoch_loss)
        losses['val'].append(val_loss)

        logging.info(f"Iou train score {total_train_iou / count}")
        test_score, _ = infer_patient(net, test_loader, device, dir_out_masks)
        print(f'infer test_score {test_score}')

        # Infer the test set
        # test_score, test_loss = eval_net(net, test_loader, device)
        scores['test'].append(test_score)
        # losses['test'].append(test_loss)
        logging.info(f"Test Score {test_score}")

        train_dsc, _ = infer_patient(net, train_loader, device, 'pred_train_masks_256')
        logging.info(f'Train DSC accuracy {train_dsc}')
        # logging.info(f'Train DSC Loss {train_loss}')

    writer.close()


    with open('results.json', 'w') as fp:
        json.dump(scores, fp)
        json.dump(losses, fp)

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    epoch_loss = 0
    criterion = dice_loss
    count = 0
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
            count += 1
            imgs, true_masks = batch["image"], batch["mask"]
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            # true_masks = true_masks / 255.

            with torch.no_grad():
                mask_pred = net(imgs)
                # mask_pred = F.softmax(mask_pred, dim=1)
                loss = criterion(mask_pred, true_masks)
                epoch_loss += loss.item()

            # if net.n_classes > 1:
            #     tot += F.cross_entropy(mask_pred, true_masks).item()
            # else:
            # pred = F.softmax(mask_pred, dim=1)
            # pred = torch.argmax(mask_pred, dim=1).float()
            dsc = dice_coeff(mask_pred, true_masks).item()
            if math.isnan(dsc):
                dsc = 0.
            tot += dsc
            pbar.update()

    net.train()
    return tot / count, epoch_loss


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """

    logging.basicConfig(filename='logger.log', filemode='w', level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=2, bilinear=True)
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
            batch_size=8,  # args.batchsize,
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

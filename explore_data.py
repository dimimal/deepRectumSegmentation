#!/usr/bin/env python3
import time
import os
import sys
import numpy as np
import cv2
from scipy import io
import matplotlib.pyplot as plt
import matplotlib
import glob
import re
from utils import clean_annotations

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
plt.rcParams["figure.figsize"] = (80, 81)

# ax = fig.add_subplot(111, projection='3d' )#, figsize=(14, 12))

SHOW = False
CT_PATH = "./data_VT1_P_5828F1K1/data_VT1_P_5828F1K1/CT"
MVCT_PATH = "./data_VT1_P_5828F1K1/data_VT1_P_5828F1K1/MVCT"


def main(args):
    print(args)
    mvct_images_paths = glob.glob(MVCT_PATH + "/**/imageT*")
    mvct_convhull_paths = glob.glob(MVCT_PATH + "/**/rectumT_man_seg_slice_convhull*")

    # Trim the annotations and images
    mvct_images_paths, mvct_convhull_paths = clean_annotations(
        mvct_images_paths, mvct_convhull_paths
    )

    # Planning images and convex slices
    # print(io.loadmat(CT_PATH + '/imageP.mat')
    ct_images = io.loadmat(CT_PATH + "/imageP.mat")["imageP"]
    ct_convex = io.loadmat(CT_PATH + "/rectumP_man_seg_slice_convhull.mat")[
        "rectumP_seg_man"
    ]
    # TODO Test MvcT the image if they have the same shape  with the annotation slices

    mvct_convhull = io.loadmat(mvct_convhull_paths[0])["rectumT_seg_man"]
    mvct_image = io.loadmat(mvct_images_paths[0])["imageT"]

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    if int(args[1]) == 0:
        print("Plot MVCT segmentations")
        print(mvct_convhull.shape)
        for i in range(mvct_convhull.shape[2]):
            # N, M = mvct_image[:, :, i].shape
            # left = N//2 - 128
            # right = N//2 + 128
            resized_mvct = cv2.resize(
                mvct_image[:, :, i], (256, 256), cv2.INTER_LANCZOS4
            )
            resized_mvct_mask = cv2.resize(
                mvct_convhull[:, :, i], (256, 256), cv2.INTER_NEAREST
            )
            ax[0].imshow(resized_mvct, cmap="gray")
            ax[1].imshow(mvct_image[:, :, i], cmap="gray")
            # ax[1].imshow(resized_mvct_mask, cmap='gray')
            plt.draw()
            plt.pause(0.1)
            # plt.show(block=False)
            print(i)
            # time.sleep(0.2)
        plt.close()

    elif int(args[1]) == 1:
        print("Plot CT segmentations")
        for i in range(ct_convex.shape[2]):
            # TODO DEBUG only
            if i < 100:
                continue
            # Upsampling
            resized_ct = cv2.resize(ct_images[:, :, i], (512, 512), cv2.INTER_LANCZOS4)
            resized_ct_mask = cv2.resize(
                ct_convex[:, :, i], (512, 512), cv2.INTER_NEAREST
            )
            ax[0].imshow(resized_ct, cmap="gray")
            # ax[1].imshow(resized_ct_mask, cmap='gray')
            ax[1].imshow(ct_images[:, :, i], cmap="gray")
            # ax[1].imshow(ct_convex[:, :, i], cmap='gray')
            plt.draw()
            plt.pause(0.1)
            print(i)
        plt.close()

    # ct_mat_3d_plan = io.loadmat(CT_PATH + '/imageP.mat')['imageP']
    # ct_mat_rectum_label_3d = io.loadmat(CT_PATH + '/rectumP_man_seg.mat')
    # print(ct_mat_3d_plan.shape)
    # print(type(ct_mat_rectum_label_3d['rectumP_man_seg']))
    # rectum_plan = ct_mat_rectum_label_3d['rectumP_man_seg']
    # print(rectum_plan[:,2])

    # plt.plot(ct_mat_3d_plan[:, 0].flatten(), ct_mat_3d_plan[:, 1].flatten(), ct_mat_3d_plan[:, 2].flatten())
    # plt.show()

    # if SHOW:
    #     # mng.full_screen_toggle()
    #     # print(matplotlib.get_backend())
    #     for i in range(ct_mat_3d_plan.shape[-1]):
    #         plt.imshow(ct_mat_3d_plan[:,:,i], cmap='gray')
    #         plt.plot(rectum_plan[:, 0], rectum_plan[:, 1], color='r')
    #         plt.show()
    #         # input()


if __name__ == "__main__":
    main(sys.argv)

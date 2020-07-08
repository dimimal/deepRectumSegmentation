import os
import sys
from scipy import io
import numpy as np
import glob
from utils import load_data, clean_annotations, export_images


CT_PATH = "./data_VT1_P_5828F1K1/**/CT"
MVCT_PATH = "./data_VT1_P_5828F1K1/**/MVCT"
OUT_CT_PATH = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/CT_1"
OUT_MVCT_PATH = "/home/dimitris/SOTON_COURSES/Msc_Thesis/Data/data/MVCT_1"


def main():
    """Main function
    :arg1: TODO
    :returns: TODO
    """
    print(os.path.exists(MVCT_PATH))
    mvct_images_paths = sorted(glob.glob(MVCT_PATH + "/**/imageT*"))
    mvct_convhull_paths = sorted(
        glob.glob(MVCT_PATH + "/**/rectumT_man_seg_slice_convhull*")
    )

    assert len(mvct_images_paths) != len(mvct_convhull_paths)
    # print(mvct_images_paths)
    print(len(mvct_images_paths))

    # Trim the annotations and images if trim is true
    mvct_images_paths, mvct_convhull_paths = clean_annotations(
        mvct_images_paths, mvct_convhull_paths, trim=False
    )

    export_images(mvct_images_paths, mvct_convhull_paths, OUT_MVCT_PATH, extract="mvct")

    # Planning images and convex slices
    # ct_images = io.loadmat(CT_PATH + '/imageP.mat')['imageP']
    # ct_masks = io.loadmat(CT_PATH + '/rectumP_man_seg_slice_convhull.mat')['rectumP_seg_man']
    # ct_images = load_data(CT_PATH+'/*.mat')
    ct_image_paths = sorted(glob.glob(CT_PATH + "/imageP*"))
    ct_mask_paths = sorted(glob.glob(CT_PATH + "/rectumP_man_seg_slice_convhull.mat"))
    # print(ct_image_paths)
    # print(ct_mask_paths)
    export_images(
        ct_image_paths,
        ct_mask_paths,
        OUT_CT_PATH,
        resize=True,
        sampling_size=(512, 512),
        keys={"image": "imageP", "mask": "rectumP_seg_man"},
        extract="ct",
    )


if __name__ == "__main__":
    main()

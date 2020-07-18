#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
from numpy import *
from mpl_toolkits.mplot3d import Axes3D

PATH = './data/MVCT_3D/mask/'


def main():
    data = glob.glob(PATH + '/**/**/*.npy')

    xx, yy = np.meshgrid(np.linspace(0, 512, 512), np.linspace(0, 512, 512))
    # print(data)
    for i in data:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xx, yy, 0)
        voxel = np.load(i)
        voxel = voxel / 255.
        indexes = np.argwhere(voxel==1)
        # print(indexes[0][0])
        # print(voxel.shape)
        for j in range(indexes.shape[0]):
            # z = indexes[j][0]
            ax.scatter(indexes[j][1], indexes[j][2], indexes[j][0], c='r')
        plt.show()
        a = input('Enter key')




if __name__ == "__main__":
    main()

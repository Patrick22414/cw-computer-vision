import time

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from numpy import ndarray as array


@jit()
def hist(im: array, MAX_L=256, density=True) -> array:
    pdf = np.zeros([MAX_L, ], dtype=np.int)
    for pixel in im.flatten():
        pdf[pixel] += 1
    if density:
        pdf = pdf / im.size

    return pdf


# @jit
def histeq(im: array, MAX_L=256) -> array:
    pdf = hist(im)

    cdf = np.zeros([MAX_L, ])
    for i in range(MAX_L):
        cdf[i] = np.round(np.sum(pdf[0:i]) * (MAX_L - 1))
    im_equalized = np.array(list(map(lambda x: cdf[x], im)), dtype=np.uint8)

    return im_equalized


if __name__ == '__main__':
    im: array = plt.imread('../cat.jpg')

    t0 = time.time()
    im_eq = histeq(im)
    t1 = time.time()
    im_eq = histeq(im)
    t2 = time.time()
    im_eq = histeq(im)
    t3 = time.time()

    print(t1 - t0)
    print(t2 - t1)
    print(t3 - t2)

    if True:
        plt.subplot(2, 2, 1)
        if im.ndim < 3:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)

        plt.subplot(2, 2, 3)
        pdf = hist(im)
        plt.bar(np.arange(pdf.size), pdf, width=1, align='edge')

        plt.subplot(2, 2, 2)
        if im.ndim < 3:
            plt.imshow(im_eq, cmap='gray')
        else:
            plt.imshow(im_eq)

        plt.subplot(2, 2, 4)
        pdf_eq = hist(im_eq)
        plt.bar(np.arange(pdf_eq.size), pdf_eq, width=1, align='edge')

        plt.show()

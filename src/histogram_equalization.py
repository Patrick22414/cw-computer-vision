import time

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from numpy import ndarray as array


# Calculate the PDF of an image
# (more accurately it's PMF)
# @jit(cache=True)
def im2pdf(im: array, MAX_L=256, normalize=True) -> array:
    pdf = np.zeros([MAX_L, ])
    for pixel in im.flatten():
        pdf[pixel] += 1
    if normalize:
        pdf = pdf / im.size

    return pdf


# A faster version of im2pdf using numpy.bincount
def im2pdf_2(im: array, MAX_L=256, normalize=True) -> array:
    pdf = np.bincount(im.flatten(), minlength=MAX_L)
    if normalize:
        pdf = pdf / im.size

    return pdf


# @jit(cache=True)
def im2cdf(im: array, MAX_L=256) -> array:
    pdf = im2pdf_2(im, MAX_L=MAX_L)
    cdf = np.zeros([MAX_L, ])
    for i in range(MAX_L):
        cdf[i] = np.floor(np.sum(pdf[0:i]) * (MAX_L - 1))

    return cdf


def histeq(im: array, MAX_L=256) -> array:
    cdf = im2cdf(im, MAX_L=MAX_L)
    im_eq = np.array(list(map(lambda x: cdf[x], im)), dtype=np.uint8)

    return im_eq


def histeq_regional(im: array, k_size, MAX_L=256) -> array:
    h = im.shape[0] - im.shape[0] % k_size
    w = im.shape[1] - im.shape[1] % k_size

    if im.ndim == 2:
        im = im[0:h, 0:w]
        im_eq = np.zeros_like(im)
        for i in range(0, h, k_size):
            for j in range(0, w, k_size):
                im_eq[i:i+k_size, j:j+k_size] = histeq(im[i:i+k_size, j:j+k_size])
    else:
        im = im[0:h, 0:w, :]
        im_eq = np.zeros_like(im)
        for i in range(0, h, k_size):
            for j in range(0, w, k_size):
                im_eq[i:i+k_size, j:j+k_size, :] = histeq(im[i:i+k_size, j:j+k_size, :])

    return im_eq


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg')

    start = time.time()

    MAX_L = 256
    color_map = 'gray' if im.ndim < 3 else None

    im_eq = histeq(im, MAX_L=MAX_L)

    if True:
        pdf = im2pdf_2(im)
        cdf = im2cdf(im)

        pdf_eq = im2pdf_2(im_eq)
        cdf_eq = im2cdf(im_eq)

        plt.subplot(2, 3, 1)
        plt.imshow(im, cmap=color_map)

        plt.subplot(2, 3, 2)
        plt.bar(range(MAX_L), pdf, width=1)

        plt.subplot(2, 3, 3)
        plt.bar(range(MAX_L), cdf, width=1)

        plt.subplot(2, 3, 4)
        plt.imshow(im_eq, cmap=color_map)

        plt.subplot(2, 3, 5)
        plt.bar(range(MAX_L), pdf_eq, width=1)

        plt.subplot(2, 3, 6)
        plt.bar(range(MAX_L), cdf_eq, width=1)

        print(time.time() - start)

        plt.show()

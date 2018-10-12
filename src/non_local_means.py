import time

import numpy as np
from matplotlib import pyplot as plt


def white_noise(im, scale):
    im = im + np.random.normal(0.0, scale, im.shape)
    im = np.maximum(im, 0.0)
    im = np.minimum(im, 1.0)
    return im


def salt_and_pepper(im, prob):
    if prob > 1 or prob < 0:
        raise ValueError("Prob must be within 0 to 1")
    if im.ndim == 2:
        im = im[:, :, np.newaxis]

    h, w, _ = im.shape
    mask = np.random.rand(h, w)
    salt = mask < (prob / 2)
    pepper = mask > (1 - prob / 2)

    im[salt, :] = 1.0
    im[pepper, :] = 0.0
    return im


def _gaussian_dist(im: np.ndarray, p1, p2, n_size):
    """
    The Guassian distance between two neighbourhood
    p1, p2: central points of each neighbourhood
    n_size: neighbourhood size
    """
    h, w, _ = im.shape
    x1, y1 = p1
    x2, y2 = p2
    r = int(n_size / 2)
    mask1 = [
        np.arange((x1 - r) if (x1 - r) > 0 else 0, (x1 + r) if (x1 + r) < h else h),
        np.arange((x1 - r) if (x1 - r) > 0 else 0, (x1 + r) if (x1 + r) < h else h)
    ]


def non_local_means(im, k_size, n_size):
    """
    k_size: kernel size, over which the average is computed.
    n_size: neighbourhood size, over which the Gaussian distance is computed.
    """


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg') / 256

    im2 = salt_and_pepper(im.copy(), 0.02)

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)

    plt.show()

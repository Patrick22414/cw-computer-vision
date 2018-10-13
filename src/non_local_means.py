import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means

import bilinear_interpolation

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


def _gaussian_weight(im, p1, p2, n_size, filtering):
    pass


def non_local_means2(im, k_size, n_size, filtering):
    if im.ndim == 2:
        im = im[..., np.newaxis]

    rad = int(k_size / 2) # kernel radius

    h, w, c = im.shape

    im_extended = np.zeros([h + k_size - 1, w + k_size - 1, c])

    xa, ya = rad, rad
    xb, yb = rad + h, rad + w

    im_extended[xa:xb, ya:yb, :] = im
    for x in range(xa, xb):
        for y in range(ya, yb):
            pass


if __name__ == '__main__':
    im = plt.imread('images/moe.jpg') / 256
    im = bilinear_interpolation.bilinear_resize(im, [480, 320, 3])
    im_noisy = white_noise(im, 0.1)

    start = time.time()
    # im_nlm = non_local_means(im_noisy, 3, 7, 0.1)
    print(time.time() - start)
    start = time.time()
    im_ski = denoise_nl_means(im_noisy, 3, 7, 0.1, multichannel=True)
    print(time.time() - start)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im_noisy)
    ax[0].axis('off')
    ax[0].set_title('noisy image')
    ax[1].imshow(im)
    ax[1].axis('off')
    ax[1].set_title('Origin')
    ax[2].imshow(im_ski)
    ax[2].axis('off')
    ax[2].set_title('NLM - skimage lib')

    plt.show()

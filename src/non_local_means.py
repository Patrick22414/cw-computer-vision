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
    """
    The Guassian distance weight between two neighbourhoods
    p1, p2: central points of each neighbourhood
    n_size: neighbourhood size
    filtering: filtering parameter
    """
    h, w, _ = im.shape
    r = int(n_size / 2)

    x, y = p1
    xa = (x - r) if (x - r) > 0 else 0
    ya = (y - r) if (y - r) > 0 else 0
    xb = (x + r + 1) if (x + r + 1) < h else h
    yb = (y + r + 1) if (y + r + 1) < w else w
    mask1 = im[xa:xb, ya:yb, :]

    x, y = p2
    xa = (x - r) if (x - r) > 0 else 0
    ya = (y - r) if (y - r) > 0 else 0
    xb = (x + r + 1) if (x + r + 1) < h else h
    yb = (y + r + 1) if (y + r + 1) < w else w
    mask2 = im[xa:xb, ya:yb, :]

    wei = (mask1.sum() / mask1.size - mask2.sum() / mask2.size) / filtering
    wei = np.exp(- wei ** 2)
    return wei


def non_local_means(im, k_size, n_size, filtering):
    """
    k_size: kernel size, over which the average is computed.
    n_size: neighbourhood size, over which the Gaussian distance is computed.
    """
    if im.ndim == 2:
        im = im[:, :, np.newaxis]

    im_ = np.zeros_like(im)

    n_row, n_col, _ = im.shape

    r = int(k_size / 2)
    timer = 0
    for row in range(n_row):
        for col in range(n_col):
            timer += 100

            row_a = (row - r) if (row - r) > 0 else 0
            col_a = (col - r) if (col - r) > 0 else 0
            row_b = (row + r + 1) if (row + r + 1) < n_row else n_row
            col_b = (col + r + 1) if (col + r + 1) < n_col else n_col

            total_weight = 0.0

            for i in range(row_a, row_b):
                for j in range(col_a, col_b):
                    weight = _gaussian_weight(im, [row, col], [i, j], n_size, filtering)
                    im_[row, col, :] += weight * im[i, j, :]
                    total_weight += weight

            print("{:6.2f}%, {:.2f}".format(timer / n_row / n_col, total_weight))
            im_[row, col, :] /= total_weight

    return im_


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg') / 256
    im = bilinear_interpolation.bilinear_resize(im, [240, 160, 3])
    im_noisy = white_noise(im, 0.1)

    start = time.time()
    im_nlm = non_local_means(im_noisy, 3, 7, 0.1)
    print(time.time() - start)
    start = time.time()
    im_ski = denoise_nl_means(im_noisy, 3, 7, 0.1, multichannel=True)
    print(time.time() - start)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im_noisy)
    ax[0].axis('off')
    ax[0].set_title('noisy image')
    ax[1].imshow(im_nlm)
    ax[1].axis('off')
    ax[1].set_title('NLM - my method')
    ax[2].imshow(im_ski)
    ax[2].axis('off')
    ax[2].set_title('NLM - skimage lib')

    plt.show()

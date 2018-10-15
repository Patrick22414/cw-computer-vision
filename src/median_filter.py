import time

import numpy as np
from matplotlib import pyplot as plt

from bilinear_interpolation import bilinear_resize
from add_noise import salt_and_pepper, white_noise

def median_filter(im, k_size):
    if im.ndim == 2:
        im = im[..., np.newaxis]

    h, w, _ = im.shape
    r = int(k_size / 2)

    im_ = im.copy()

    count_total = h * w / 100
    count = 1

    for x in range(h):
        for y in range(w):
            xa = (x-r) if (x-r) > 0 else 0
            ya = (y-r) if (y-r) > 0 else 0
            xb = (x+r+1) if (x+r+1) < h else h
            yb = (y+r+1) if (y+r+1) < w else w

            kernel = im[xa:xb, ya:yb, :]
            median = np.median(np.median(kernel, axis=0), axis=0)

            im_[x, y, :] = median

            count += 1
            if count % 800 == 0:
                print("{:.0f}%".format(count / count_total))

    return np.squeeze(im_)


if __name__ == '__main__':
    im = plt.imread('images/moe2d.jpg') / 256

    # im = bilinear_resize(im, [240, 160, 3])

    im_noisy = salt_and_pepper(im, 0.1)
    # im_noisy = white_noise(im, 0.1)

    im_mf = median_filter(im_noisy, 3)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Origin')

    ax[1].imshow(im_noisy, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Noisy')

    ax[2].imshow(im_mf, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title('Median filter')

    plt.show()

import numpy as np
from matplotlib import pyplot as plt

from add_noise import salt_and_pepper


def median_filter(im, k_size):
    if im.ndim == 2:
        im = im[..., np.newaxis]

    h, w, _ = im.shape
    r = int(k_size / 2)

    im_ = im.copy()

    for x in range(h):
        print(f'{x}/{h}')
        for y in range(w):
            xa = (x-r) if (x-r) > 0 else 0
            ya = (y-r) if (y-r) > 0 else 0
            xb = (x+r+1) if (x+r+1) < h else h
            yb = (y+r+1) if (y+r+1) < w else w
            kernel = im[xa:xb, ya:yb, :]

            median = np.median(np.median(kernel, axis=0), axis=0)

            im_[x, y, :] = median

    return np.squeeze(im_)


if __name__ == '__main__':
    im = plt.imread('images/box.jpg') / 256

    im_noisy = salt_and_pepper(im, 0.1)

    im_mf = median_filter(im_noisy, 3)

    # plotting
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(16, 8)
    fig.set_tight_layout(True)
    for a in ax:
        a.axis('off')

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

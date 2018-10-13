import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def arr_resize(arr, new_size, axis=0):
    shape = list(arr.shape)
    old_size = shape[axis]
    shape[axis] = new_size

    res = np.zeros(shape)
    for i in range(new_size):
        pos = i * old_size / new_size
        low = int(pos)
        if axis == 0:
            try:
                vec_l = arr[low, :, :]
                vec_u = arr[low + 1, :, :]
                res[i, :, :] = (pos - low) * (vec_u - vec_l) + vec_l
            except IndexError:
                res[i, :] = arr[low, :, :]
        elif axis == 1:
            try:
                vec_l = arr[:, low, :]
                vec_u = arr[:, low + 1, :]
                res[:, i, :] = (pos - low) * (vec_u - vec_l) + vec_l
            except IndexError:
                res[:, i, :] = arr[:, low, :]
        else:
            raise ValueError("The axis argument can be either 0 or 1")

    return res


def bilinear_resize(im, new_shape):
    if im.dtype != np.float64:
        raise TypeError("Image must be of type numpy.float64")
    if im.ndim != len(new_shape):
        raise Warning("The new shape and the image are of different ndim")

    if im.ndim == 2:
        im = im[:, :, np.newaxis]

    tmp = arr_resize(im, new_shape[0], axis=0)
    res = arr_resize(tmp, new_shape[1], axis=1)

    return np.squeeze(res)


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg') / 256
    start = time.time()
    # im_resized = bilinear_resize(im, [640, 960, 3])
    for _ in range(100):
        im_resized = bilinear_resize(im, [640, 960, 3])
    print(time.time() - start)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(im_resized, cmap='gray')
    ax[1].axis('off')

    plt.show()

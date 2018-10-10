import time

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
                vec_l = arr[low, :]
                vec_u = arr[low + 1, :]
                res[i, :] = (pos - low) * (vec_u - vec_l) + vec_l
            except IndexError:
                res[i, :] = arr[low, :]
        elif axis == 1:
            try:
                vec_l = arr[:, low]
                vec_u = arr[:, low + 1]
                res[:, i] = (pos - low) * (vec_u - vec_l) + vec_l
            except IndexError:
                res[:, i] = arr[:, low]
        else:
            raise ValueError

    return res


def bilinear_resize(im, new_shape):
    if im.dtype != np.float64:
        im = im.astype(np.float64)

    if im.ndim == 2:
        tmp = arr_resize(im, new_shape[0], axis=0)
        res = arr_resize(tmp, new_shape[1], axis=1)
        return res
    elif im.ndim == 3:
        res = np.zeros(new_shape)
        for n, ch in enumerate(im):
            print(n, ch.shape)
            # tmp = arr_resize(ch, new_shape[0], axis=0)
            # res[:, :, n] = arr_resize(tmp, new_shape[1], axis=1)
        return res
    else:
        raise ValueError(f"Image ndim can only be 2 or 3, yet input ndim is {im.ndim}")


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg')
    start = time.time()
    im_resized = bilinear_resize(im, [400, 600, 3])
    print(time.time() - start)

    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(im_resized, cmap='gray')

    plt.show()

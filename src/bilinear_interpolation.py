import time

import numpy as np
from matplotlib import pyplot as plt


def arr_resize(arr, new_size, axis=0):
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)

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



if __name__ == '__main__':
    im = plt.imread('images/HMS_Implacable.jpg')
    start = time.time()
    for _ in range(100):
        im_resized = arr_resize(im, 400, axis=0)
        im_resized = arr_resize(im_resized, 600, axis=1)
    print(time.time() - start)

    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(im_resized, cmap='gray')

    plt.show()

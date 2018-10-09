import numpy as np
from matplotlib import pyplot as plt


def vec_resize(vec, size_out):
    size_in = vec.size
    res = np.zeros([size_out, ])

    for i in range(size_out):
        position = i * size_in / size_out
        lower = int(position)
        upper = (lower + 1) if (lower + 1) < size_in else lower

        res[i] = (position - lower) * (vec[upper] - vec[lower]) + vec[lower]

    return res


im = plt.imread('images/HMS_Implacable.jpg')
h, w = im.shape
h2, w2 = 400, 600

im_intermedia = np.zeros([h, w2])
im_resized = np.zeros([h2, w2])
for i in range(h):
    im_intermedia[i, :] = vec_resize(im[i, :], w2)
for j in range(w2):
    im_resized[:, j] = vec_resize(im_intermedia[:, j], h2)

print(im.dtype)
print(im_resized.dtype)
plt.imshow(im_resized, cmap='gray')
plt.show()

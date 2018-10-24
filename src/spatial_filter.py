import numpy as np
from matplotlib import pyplot as plt


def conv(im: np.ndarray, kernel: np.ndarray, pad='zero'):
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel must have odd number of rows and cols")

    if im.ndim == 2 and kernel.ndim == 2:
        h, w = im.shape
        kh, kw = kernel.shape

        if pad == 'none':
            im_pad = im.copy()
            im_res = np.zeros([h - kh + 1, w - kw + 1])
        elif pad == 'zero':
            im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2)], 'constant', constant_values=0)
            im_res = np.zeros_like(im)
        elif pad == 'edge':
            im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2)], 'edge')
            im_res = np.zeros_like(im)
        else:
            raise NotImplementedError("Currently available padding method: 'none', 'zero', 'edge'")

        start = [(x, y) for x in range(h) for y in range(w)]
        end = [(x, y) for x in range(kh, h + kh) for y in range(kw, w + kw)]

        for s, e in zip(start, end):
            im_res[s] = np.sum(im_pad[s[0]:e[0], s[1]:e[1]] * kernel)

        return im_res

    elif im.ndim == 3 and kernel.ndim == 2:
        h, w, c = im.shape
        kh, kw = kernel.shape

        kernel = np.stack([kernel] * c, axis=2)

        if pad == 'none':
            im_pad = im.copy()
            im_res = np.zeros([h - kh + 1, w - kw + 1])
        elif pad == 'zero':
            im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)], 'constant', constant_values=0)
            im_res = np.zeros_like(im)
        elif pad == 'edge':
            im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)], 'edge')
            im_res = np.zeros_like(im)
        else:
            raise NotImplementedError("Currently available padding method: 'none', 'zero', 'edge'")

        start = [(x, y) for x in range(h) for y in range(w)]
        end = [(x, y) for x in range(kh, h + kh) for y in range(kw, w + kw)]

        for s, e in zip(start, end):
            patch = im_pad[s[0]:e[0], s[1]:e[1], :] * kernel
            im_res[s] = np.sum(np.sum(patch, axis=0), axis=0)

        return im_res

    elif im.ndim == 3 and kernel.ndim == 3:
        pass
        # if im.shape[2] != kernel.shape[2]:
        #     raise ValueError("Image and kernel do not match in number of channels")
        # return _conv_mc(im, kernel, pad)

    else:
        raise ValueError("Image or kernel is not of valid number of dimensions")


if __name__ == '__main__':
    im = plt.imread('F:\Documents\GitHub\cw-computer-vision\images\moe2d.jpg') / 256
    kernel = np.ones([5, 5])
    kernel = kernel / np.sum(kernel)

    im_1 = conv(im, kernel, 'zero')

    plt.imshow(im_1)
    plt.show()

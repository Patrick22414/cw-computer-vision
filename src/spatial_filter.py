import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d


def conv(im: np.ndarray, kernel: np.ndarray, pad='zero'):
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel must have odd number of rows and cols")

    if pad not in ['zero', 'none', 'edge']:
        raise ValueError(
            "Currently available padding method: 'none', 'zero', 'edge'")

    if im.ndim == 2 and kernel.ndim == 2:
        return _conv_1c(im, kernel, pad)

    elif im.ndim == 3 and kernel.ndim == 2:
        kernel = np.stack([kernel] * im.shape[-1], axis=2)
        return _conv_mc(im, kernel, pad)

    elif im.ndim == 3 and kernel.ndim == 3:
        if im.shape[2] != kernel.shape[2]:
            raise ValueError(
                "Image and kernel do not match in number of channels")
        return _conv_mc(im, kernel, pad)

    else:
        raise ValueError(
            "Image or kernel is not of valid number of dimensions")


# single channel convolution
def _conv_1c(im, kernel, pad):
    h, w = im.shape
    kh, kw = kernel.shape

    if pad == 'none':
        im_pad = im.copy()
        im_res = np.zeros([h - kh + 1, w - kw + 1])
    elif pad == 'zero':
        im_pad = np.pad(
            im, [(kh // 2, kh // 2), (kw // 2, kw // 2)], 'constant', constant_values=0)
        im_res = np.zeros_like(im)
    elif pad == 'edge':
        im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2)], 'edge')
        im_res = np.zeros_like(im)
    else:
        raise Exception

    start = [(x, y) for x in range(h) for y in range(w)]
    end = [(x, y) for x in range(kh, h + kh) for y in range(kw, w + kw)]

    for s, e in zip(start, end):
        im_res[s] = np.sum(im_pad[s[0]:e[0], s[1]:e[1]] * kernel)

    return im_res


# multi-channel convolution
def _conv_mc(im, kernel, pad):
    h, w, _ = im.shape
    kh, kw, _ = kernel.shape

    if pad == 'none':
        im_pad = im.copy()
        im_res = np.zeros([h - kh + 1, w - kw + 1])
    elif pad == 'zero':
        im_pad = np.pad(im, [(kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)], 'constant', constant_values=0)
        im_res = np.zeros_like(im)
    elif pad == 'edge':
        im_pad = np.pad(
            im, [(kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0)], 'edge')
        im_res = np.zeros_like(im)
    else:
        raise Exception

    start = [(x, y) for x in range(h) for y in range(w)]
    end = [(x, y) for x in range(kh, h + kh) for y in range(kw, w + kw)]

    for s, e in zip(start, end):
        patch = im_pad[s[0]:e[0], s[1]:e[1], :] * kernel
        im_res[s] = np.sum(np.sum(patch, axis=0), axis=0)

    return im_res


# normalise the image to 0-1
def normalise(im):
    im = im - np.min(im)
    im = im / (np.max(im) + 1e-9)
    return im


if __name__ == '__main__':
    im = pyplot.imread('images/moon.jpg') / 256

    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    uniform = np.ones([5, 5]) / 15

    im_b = convolve2d(im, laplace, mode='same')
    im_c = np.clip(im + im_b, 0.0, 1.0)
    im_d = np.abs(convolve2d(im, sobel_x, mode='same') + convolve2d(im, sobel_y, mode='same'))
    im_e = convolve2d(im_d, uniform, mode='same')
    im_f = normalise(im_c * im_e)
    im_g = normalise(im + im_f)
    im_h = np.power(im_g, 0.7)

    # plotting
    cmap = 'gray'
    fontsize = 16

    fig, ax = pyplot.subplots(2, 4)
    fig.set_size_inches(16, 9)
    fig.set_tight_layout(True)
    for x in range(2):
        for y in range(4):
            ax[x, y].get_xaxis().set_visible(False)
            ax[x, y].get_yaxis().set_visible(False)

    ax[0, 0].imshow(im, cmap=cmap)
    ax[0, 0].set_title('Origin (a)', fontsize=fontsize)

    ax[0, 1].imshow(normalise(im_b), cmap=cmap)
    ax[0, 1].set_title('Laplace (b)', fontsize=fontsize)

    ax[0, 2].imshow(im_c, cmap=cmap)
    ax[0, 2].set_title('Laplace sharpen (c)', fontsize=fontsize)

    ax[0, 3].imshow(im_d, cmap=cmap)
    ax[0, 3].set_title('Sobel (d)', fontsize=fontsize)

    ax[1, 0].imshow(im_e, cmap=cmap)
    ax[1, 0].set_title('Sobel smooth (e)', fontsize=fontsize)

    ax[1, 1].imshow(im_f, cmap=cmap)
    ax[1, 1].set_title('Sobel mask (f)', fontsize=fontsize)

    ax[1, 2].imshow(im_g, cmap=cmap)
    ax[1, 2].set_title('Sobel sharpen (g)', fontsize=fontsize)

    ax[1, 3].imshow(im_h, cmap=cmap)
    ax[1, 3].set_title('Sobel final (h)', fontsize=fontsize)

    pyplot.show()

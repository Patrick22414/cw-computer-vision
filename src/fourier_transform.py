import numpy as np
from matplotlib import pyplot


def dft_2d(im: np.ndarray):
    f = np.zeros_like(im, dtype=np.complex128)
    m, n = im.shape
    c = -2j * np.pi
    g = np.zeros_like(im, dtype=np.float64)
    for u in range(m):
        for v in range(n):
            for x in range(m):
                for y in range(n):
                    g[x, y] = u * x / m + v * y / n
            f[u, v] = np.sum(im * np.exp(c * g))

    return f


def dft_1d(vec: np.ndarray):
    f = np.zeros_like(vec, dtype=np.complex128)
    n = vec.size
    g = -2j * np.pi / n * np.arange(n, dtype=np.complex128)
    for k in range(n):
        f[k] = np.sum(vec * np.exp(g * k))

    return f


def dft_2d_by_1d(im: np.ndarray):
    m, n = im.shape
    f = np.zeros_like(im, dtype=np.complex128)
    g = np.zeros_like(im, dtype=np.complex128)
    for u in range(m):
        g[u, :] = dft_1d(im[u, :])
    for v in range(n):
        f[:, v] = dft_1d(g[:, v])

    return f


if __name__ == '__main__':
    im = pyplot.imread('../images/zebra_line.jpg') / 256

    # shift fourier transform
    im_s = im.copy()
    im_s[1::2, ::2] *= -1
    im_s[::2, 1::2] *= -1

    im_f = dft_2d_by_1d(im_s)

    # augment output
    im_f = np.abs(im_f)
    im_f = np.log(im_f)
    im_f = im_f / np.max(im_f)

    # plotting
    fig, ax = pyplot.subplots(1, 2)
    fig.set_size_inches(12, 8)
    fig.set_tight_layout(True)
    for a in ax:
        a.axis('off')

    ax[0].imshow(im, cmap='gray')
    ax[0].set_title('Origin', fontsize=16)

    ax[1].imshow(im_f, cmap='gray')
    ax[1].set_title('Fourier transform', fontsize=16)

    pyplot.show()

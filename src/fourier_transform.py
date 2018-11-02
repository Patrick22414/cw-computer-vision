import numpy as np
import time


def dft_2d(im: np.ndarray):
    f = np.zeros_like(im, dtype=np.complex128)
    m, n = im.shape
    twopijg = -2j * np.pi
    for u in range(m):
        for v in range(n):
            g = np.zeros_like(im, dtype=np.float64)
            for x in range(m):
                for y in range(n):
                    g[x, y] = u * x / m + v * y / n
            g = np.exp(twopijg * g)
            f[u, v] = np.sum(im * g)

    return f


if __name__ == '__main__':
    # im = pyplot.imread('../images/HMS.jpg') / 256
    im = np.random.rand(80, 80)
    start = time.time()
    im_f = dft_2d(im)
    print(time.time() - start)

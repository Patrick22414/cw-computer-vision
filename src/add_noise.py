import numpy as np


def white_noise(im, scale):
    im = im + np.random.normal(0.0, scale, im.shape)
    im = np.maximum(im, 0.0)
    im = np.minimum(im, 1.0)
    return im


def salt_and_pepper(im, prob):
    if prob > 1 or prob < 0:
        raise ValueError("Prob must be within 0 to 1")
    if im.ndim == 2:
        im = im[:, :, np.newaxis]

    h, w, _ = im.shape
    mask = np.random.rand(h, w)
    salt = mask < (prob / 2)
    pepper = mask > (1 - prob / 2)

    im_ = im.copy()

    im_[salt, :] = 1.0
    im_[pepper, :] = 0.0
    return im_

import time

import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma

import bilinear_interpolation
from add_noise import salt_and_pepper, white_noise


def _gaussian_weight(im, p1, p2, n_size, filtering):
    r = int(n_size / 2)

    x1, y1 = p1
    x2, y2 = p2

    area1 = im[
        (x1 - r):(x1 + r + 1),
        (y1 - r):(y1 + r + 1),
        :
    ]
    area2 = im[
        (x2 - r):(x2 + r + 1),
        (y2 - r):(y2 + r + 1),
        :
    ]

    distance = np.sum(np.square(area1 - area2) - filtering)
    if distance > 5:
        return 0.0
    else:
        return np.exp(-max(distance, 0))


def non_local_means(im, k_size, n_size, filtering):
    if im.ndim == 2:
        im = im[..., np.newaxis]

    r = int(k_size / 2)  # kernel radius

    h, w, c = im.shape

    im_extended = np.zeros([h + k_size - 1, w + k_size - 1, c])
    im_result = np.zeros_like(im_extended)

    xa, ya = r, r
    xb, yb = r + h, r + w

    im_extended[xa:xb, ya:yb, :] = im

    # count_total = h * w / 100
    # count = 0

    for x in range(xa, xb):
        for y in range(ya, yb):
            xxa = (x - r) if (x - r) > xa else xa
            xxb = (x + r + 1) if (x + r + 1) < xb else xb
            yya = (y - r) if (y - r) > ya else ya
            yyb = (y + r + 1) if (y + r + 1) < yb else yb

            weight_total = 0.0

            for xx in range(xxa, xxb):
                for yy in range(yya, yyb):
                    weight = _gaussian_weight(
                        im_extended,
                        [x, y],
                        [xx, yy],
                        n_size,
                        filtering
                    )
                    # print(weight)
                    weight_total += weight
                    im_result[x, y, :] += weight * im_extended[xx, yy, :]

            im_result[x, y, :] /= weight_total

            # print("{:6.2f}%".format(count / count_total))
            # count += 1

    return np.squeeze(im_result[xa:xb, ya:yb, :])


if __name__ == '__main__':
    im = plt.imread('images/cat.jpg') / 256
    # im = bilinear_interpolation.bilinear_resize(im, [120, 90, 3])
    im_noisy = white_noise(im, 0.1)

    print(estimate_sigma(im_noisy, average_sigmas=True, multichannel=True))

    # start = time.time()
    # im_nlm = non_local_means(im_noisy, 5, 3, 0.1)
    # print(time.time() - start)

    # start = time.time()
    # im_ski = denoise_nl_means(im_noisy, 5, 3, 0.1, multichannel=True)
    # print(time.time() - start)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(im_noisy, cmap='gray')
    # ax[0].axis('off')
    # ax[0].set_title('noisy image')
    # ax[1].imshow(im_nlm, cmap='gray')
    # ax[1].axis('off')
    # ax[1].set_title('NLM - self-built')
    # ax[2].imshow(im_ski, cmap='gray')
    # ax[2].axis('off')
    # ax[2].set_title('NLM - skimage lib')

    # plt.show()

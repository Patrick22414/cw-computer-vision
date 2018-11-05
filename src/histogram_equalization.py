import numpy as np
from matplotlib import pyplot
from numpy import ndarray as array


# Calculate the PDF/PMF of an image
def im2pdf(im: array, MAX_L=256, normalize=True) -> array:
    pdf = np.zeros([MAX_L, ])
    for pixel in im.flatten():
        pdf[pixel] += 1
    if normalize:
        pdf = pdf / im.size

    return pdf


# A faster version of im2pdf using numpy.bincount
def im2pdf_2(im: array, MAX_L=256, normalize=True) -> array:
    pdf = np.bincount(im.flatten(), minlength=MAX_L)
    if normalize:
        pdf = pdf / im.size

    return pdf


# Calculate the CDF of an image
def im2cdf(im: array, MAX_L=256) -> array:
    pdf = im2pdf_2(im, MAX_L=MAX_L)
    cdf = np.zeros([MAX_L, ])
    for i in range(MAX_L):
        cdf[i] = np.floor(np.sum(pdf[0:i]) * (MAX_L - 1))

    return cdf


def histeq(im: array, MAX_L=256) -> array:
    cdf = im2cdf(im, MAX_L=MAX_L)
    im_eq = np.array(list(map(lambda x: cdf[x], im)), dtype=np.uint8)

    return im_eq


if __name__ == '__main__':
    im = pyplot.imread('images/cat.jpg')
    MAX_L = 256

    im_eq = histeq(im, MAX_L=MAX_L)

    # get PDF, CDF of each image
    pdf = im2pdf_2(im)
    cdf = im2cdf(im)

    pdf_eq = im2pdf_2(im_eq)
    cdf_eq = im2cdf(im_eq)

    # plotting
    fig, ax = pyplot.subplots(2, 3)
    fig.set_size_inches(12, 8)
    fig.set_tight_layout(True)

    ax[0, 0].imshow(im, cmap='gray')
    ax[0, 0].set_title('Origin')
    ax[0, 0].axis('off')

    ax[0, 1].bar(range(MAX_L), pdf, width=1)
    ax[0, 1].set_xlim([0, MAX_L-1])

    ax[0, 2].bar(range(MAX_L), cdf, width=1)
    ax[0, 2].set_xlim([0, MAX_L-1])

    ax[1, 0].imshow(im_eq, cmap='gray')
    ax[1, 0].set_title('Histogram equalized')
    ax[1, 0].axis('off')

    ax[1, 1].bar(range(MAX_L), pdf_eq, width=1)
    ax[1, 1].set_xlim([0, MAX_L-1])

    ax[1, 2].bar(range(MAX_L), cdf_eq, width=1)
    ax[1, 2].set_xlim([0, MAX_L-1])

    pyplot.show()

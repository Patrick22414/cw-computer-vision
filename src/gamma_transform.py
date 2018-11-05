import numpy as np
from matplotlib import pyplot

if __name__ == '__main__':
    filename = '../images/box.jpg'
    im = pyplot.imread(filename) / 256

    im_02 = np.power(im, 0.2)
    im_05 = np.power(im, 0.5)
    im_15 = np.power(im, 1.5)
    im_20 = np.power(im, 2.0)
    im_30 = np.power(im, 3.0)

    # plotting
    fig, ax = pyplot.subplots(2, 3)
    fig.set_size_inches(10, 8)
    fig.set_tight_layout(True)
    for axis in ax.flatten():
        axis.axis('off')

    ax[0, 0].imshow(im, cmap='gray')
    ax[0, 0].set_title('Origin', fontsize=16)

    ax[0, 1].imshow(im_02, cmap='gray')
    ax[0, 1].set_title('Gamma=0.2', fontsize=16)

    ax[0, 2].imshow(im_05, cmap='gray')
    ax[0, 2].set_title('Gamma=0.5', fontsize=16)

    ax[1, 0].imshow(im_15, cmap='gray')
    ax[1, 0].set_title('Gamma=1.5', fontsize=16)

    ax[1, 1].imshow(im_20, cmap='gray')
    ax[1, 1].set_title('Gamma=2.0', fontsize=16)

    ax[1, 2].imshow(im_30, cmap='gray')
    ax[1, 2].set_title('Gamma=3.0', fontsize=16)

    pyplot.show()

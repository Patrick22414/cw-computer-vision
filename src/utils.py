import matplotlib
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

matplotlib.use('agg')

if __name__ == '__main__':
    file = 'images/histeq.png'

    img = Image.open(file)
    img = img.convert('L')

    # im = np.array(img, dtype=np.float64)
    # im = im[:, 15:615]
    #
    # kernel = np.ones([9, 9]) / 81
    # im = convolve2d(im, kernel, mode='same').astype(np.uint8)

    # img = Image.fromarray(im)
    img.save('results/histeq.jpg')

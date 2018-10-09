import numpy as py
from matplotlib import pyplot as plt

# def avgpool(im, k_size=2):
#     shape = im.shape
#     im = im[0:(shape - (shape[0] % k_size)), 0:(shape - (shape[1] % k_size)), :]

#     im_0 = im[0:k_size:-1, 0:k_size:-1, :]
#     im_1 = im[0:k_size:-1, 0:k_size:-1, :]
#     im_2 = im[0:k_size:-1, 0:k_size:-1, :]
#     im_3 = im[0:k_size:-1, 0:k_size:-1, :]

im = plt.imread('images/cat.jpg')

k_size = 3
print(im.shape)

print(im.shape[1] % k_size)

shape = im.shape

im = im[0:(shape[0] - shape[0] % k_size), 0:(shape[1] - shape[1] % k_size), :]
print(im.shape)

import time
import asyncio

import numpy as np
import matplotlib.pyplot as plt


def main(a: np.ndarray):
    b = np.pad(a, 3, 'constant', constant_values=0)
    b[0, 0] = 100
    a = b


if __name__ == '__main__':
    a = np.arange(12).reshape(3, 4)
    main(a)
    print(a)

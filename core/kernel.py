"""Module core.kernel - Noyaux de convolution."""

import numpy as np


class Kernel:
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float64)
        self.size = data.shape

    @classmethod
    def mean(cls, size: int = 3):
        data = np.ones((size, size), dtype=np.float64) / (size * size)
        return cls(data)

    @classmethod
    def gaussian(cls, size: int = 3, sigma: float = 1.0):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return cls(kernel / np.sum(kernel))

    @classmethod
    def sobel_x(cls):
        return cls(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))

    @classmethod
    def sobel_y(cls):
        return cls(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))

    @classmethod
    def prewitt_x(cls):
        return cls(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64))

    @classmethod
    def prewitt_y(cls):
        return cls(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64))

    @classmethod
    def roberts_x(cls):
        return cls(np.array([[1, 0], [0, -1]], dtype=np.float64))

    @classmethod
    def roberts_y(cls):
        return cls(np.array([[0, 1], [-1, 0]], dtype=np.float64))

    @classmethod
    def laplacian(cls, variant='4-connected'):
        if variant == '4-connected':
            return cls(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64))
        else:
            return cls(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64))
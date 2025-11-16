"""Module core.histogram - Gestion des histogrammes."""

import numpy as np


class Histogram:
    def __init__(self, values: np.ndarray, bins: int = 256):
        self.values = values.astype(np.float64)
        self.bins = bins

    @classmethod
    def from_image(cls, image):
        bins = int(image.max_value) + 1
        hist, _ = np.histogram(image.data.flatten(), bins=bins, range=(0, image.max_value))
        return cls(hist, bins)

    def normalize(self):
        total = self.values.sum()
        if total == 0:
            return Histogram(self.values.copy(), self.bins)
        normalized = self.values / total
        return Histogram(normalized, self.bins)

    def cumulative(self):
        return np.cumsum(self.values)

    def cumulative_normalized(self):
        cdf = self.cumulative()
        return cdf / cdf[-1] if cdf[-1] > 0 else cdf

    def mean(self):
        levels = np.arange(self.bins)
        normalized = self.normalize()
        return np.sum(levels * normalized.values)

    def equalize_lut(self):
        cdf = self.cumulative_normalized()
        return np.round(cdf * (self.bins - 1)).astype(np.uint8)
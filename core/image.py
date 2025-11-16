"""Module core.image - Classe principale pour la gestion des images."""
from utils.io_handlers import IOHandler
from core.histogram import Histogram
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class Image:
    """Classe représentant une image en niveaux de gris."""

    def __init__(self, data: np.ndarray, max_value: int = 255):
        if not isinstance(data, np.ndarray):
            raise ValueError("Les données doivent être un tableau numpy")
        if data.ndim != 2:
            raise ValueError("L'image doit être en 2D (niveaux de gris)")

        self.data = data.astype(np.float64)
        self.height, self.width = data.shape
        self.max_value = max_value

    @classmethod
    def from_file(cls, filepath: str) -> 'Image':
        return Image(*IOHandler.load_image(filepath))

    @classmethod
    def from_array(cls, array: np.ndarray, max_value: int = 255) -> 'Image':
        return cls(array, max_value)

    @classmethod
    def zeros(cls, height: int, width: int, max_value: int = 255) -> 'Image':
        data = np.zeros((height, width), dtype=np.float64)
        return cls(data, max_value)

    @classmethod
    def ones(cls, height: int, width: int, max_value: int = 255) -> 'Image':
        data = np.ones((height, width), dtype=np.float64) * max_value
        return cls(data, max_value)

    def copy(self) -> 'Image':
        return Image(self.data.copy(), self.max_value)

    def to_file(self, filepath: str, format: str = 'pgm'):
        IOHandler.save_image(self, filepath, format)

    def clip(self, min_val: float = 0, max_val: Optional[float] = None) -> 'Image':
        if max_val is None:
            max_val = self.max_value
        clipped_data = np.clip(self.data, min_val, max_val)
        return Image(clipped_data, self.max_value)

    def normalize(self, new_min: float = 0, new_max: float = 255) -> 'Image':
        old_min = self.data.min()
        old_max = self.data.max()
        if old_max - old_min == 0:
            return self.copy()
        normalized = (self.data - old_min) / (old_max - old_min)
        normalized = normalized * (new_max - new_min) + new_min
        return Image(normalized, int(new_max))

    def get_histogram(self) -> 'Histogram':
        return Histogram.from_image(self)

    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def __repr__(self) -> str:
        return f"Image(shape={self.shape()}, max_value={self.max_value})"

    def __add__(self, other) -> 'Image':
        if isinstance(other, Image):
            if self.shape() != other.shape():
                raise ValueError("Les images doivent avoir la même taille")
            result = self.data + other.data
        else:
            result = self.data + other
        return Image(np.clip(result, 0, self.max_value), self.max_value)

    def __sub__(self, other) -> 'Image':
        if isinstance(other, Image):
            if self.shape() != other.shape():
                raise ValueError("Les images doivent avoir la même taille")
            result = self.data - other.data
        else:
            result = self.data - other
        return Image(np.clip(result, 0, self.max_value), self.max_value)

    def __mul__(self, other) -> 'Image':
        result = self.data * other
        return Image(np.clip(result, 0, self.max_value), self.max_value)
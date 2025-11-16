"""Module processing.spatial - Filtrage spatial et convolution."""
from core.kernel import Kernel
from core.image import Image
import numpy as np
from scipy import signal, ndimage


class SpatialProcessor:
    """Processeur pour le filtrage spatial (convolution)."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def convolve(self, kernel, mode: str = 'same', boundary: str = 'reflect') -> 'Image':
        """
        Convolution 2D de l'image avec un noyau.

        Args:
            kernel: Instance de Kernel ou tableau numpy
            mode: Mode de convolution ('same', 'valid', 'full')
            boundary: Gestion des bords ('reflect', 'constant', 'wrap')
        """

        if isinstance(kernel, Kernel):
            kernel_data = kernel.data
        else:
            kernel_data = kernel

        # Utiliser scipy pour la convolution optimisée
        if mode == 'same':
            result = ndimage.convolve(self.image.data, kernel_data, mode=boundary)
        else:
            result = signal.convolve2d(self.image.data, kernel_data, mode=mode, boundary=boundary)

        return Image(np.clip(result, 0, self.image.max_value), self.image.max_value)

    def mean_filter(self, size: int = 3) -> 'Image':
        """
        Filtre moyenneur (lissage).

        Args:
            size: Taille du filtre (size x size)
        """
        kernel = Kernel.mean(size)
        return self.convolve(kernel)

    def gaussian_filter(self, size: int = 3, sigma: float = 1.0) -> 'Image':
        """
        Filtre gaussien (lissage).

        Args:
            size: Taille du filtre (doit être impair)
            sigma: Écart-type de la gaussienne
        """
        kernel = Kernel.gaussian(size, sigma)
        return self.convolve(kernel)

    def median_filter(self, size: int = 3) -> 'Image':
        """
        Filtre médian (non-linéaire, bon pour le bruit poivre et sel).

        Args:
            size: Taille de la fenêtre (size x size)
        """

        # Utiliser scipy pour le filtre médian optimisé
        result = ndimage.median_filter(self.image.data, size=size)

        return Image(result, self.image.max_value)

    def custom_filter(self, kernel_data: np.ndarray) -> 'Image':
        """
        Applique un filtre personnalisé.

        Args:
            kernel_data: Tableau numpy 2D représentant le noyau
        """
        kernel = Kernel(kernel_data)
        return self.convolve(kernel)
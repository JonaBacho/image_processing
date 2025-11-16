"""Module processing.pointwise - Traitements ponctuels et amélioration du contraste."""

import numpy as np
from typing import Optional, List, Tuple
from core.image import Image
from core.histogram import Histogram


class PointwiseProcessor:
    """Processeur pour les transformations ponctuelles (pixel par pixel)."""

    def __init__(self, image):
        """Initialise le processeur avec une image."""
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def linear_transform(self, s_min: Optional[float] = None, s_max: Optional[float] = None) -> 'Image':
        """
        Transformation linéaire avec saturation optionnelle.
        Étire la dynamique de l'image entre [s_min, s_max] vers [0, max_value].
        """

        data = self.image.data

        # Si pas de saturation, utiliser min/max de l'image
        if s_min is None:
            s_min = data.min()
        if s_max is None:
            s_max = data.max()

        # Éviter division par zéro
        if s_max - s_min == 0:
            return self.image.copy()

        # Transformation linéaire
        result = (data - s_min) / (s_max - s_min) * self.image.max_value
        result = np.clip(result, 0, self.image.max_value)

        return Image(result, self.image.max_value)

    def piecewise_linear(self, points: List[Tuple[float, float]]) -> 'Image':
        """
        Transformation linéaire par morceaux.

        Args:
            points: Liste de tuples (x, y) définissant les points de contrôle
                   Exemple: [(0, 0), (100, 50), (200, 255)]
        """

        if len(points) < 2:
            raise ValueError("Au moins 2 points sont nécessaires")

        # Trier les points par x
        points = sorted(points, key=lambda p: p[0])

        # Créer la LUT
        lut = np.zeros(int(self.image.max_value) + 1)

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Interpolation linéaire entre les deux points
            for x in range(int(x1), int(x2) + 1):
                if x2 - x1 != 0:
                    lut[x] = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                else:
                    lut[x] = y1

        # Étendre aux extrémités
        lut[:int(points[0][0])] = points[0][1]
        lut[int(points[-1][0]):] = points[-1][1]

        # Appliquer la LUT
        result = lut[self.image.data.astype(int)]
        return Image(result, self.image.max_value)

    def gamma_correction(self, gamma: float, k: float = 1.0) -> 'Image':
        """
        Correction gamma: I_out = k * (I_in / max)^gamma * max

        Args:
            gamma: Exposant gamma (< 1 éclaircit, > 1 assombrit)
            k: Constante multiplicative
        """
        # Normaliser à [0, 1]
        normalized = self.image.data / self.image.max_value

        # Appliquer gamma
        corrected = k * np.power(normalized, gamma)

        # Ramener à [0, max_value]
        result = corrected * self.image.max_value
        result = np.clip(result, 0, self.image.max_value)

        return Image(result, self.image.max_value)

    def histogram_equalization(self) -> 'Image':
        """Égalisation globale de l'histogramme."""

        # Obtenir l'histogramme
        hist = self.image.get_histogram()

        # Calculer la LUT d'égalisation
        lut = hist.equalize_lut()

        # Appliquer la LUT
        result = lut[self.image.data.astype(int)]
        return Image(result, self.image.max_value)

    def local_histogram_equalization(self, window_size: int = 7) -> 'Image':
        """
        Égalisation locale de l'histogramme.

        Args:
            window_size: Taille de la fenêtre locale (doit être impaire)
        """

        if window_size % 2 == 0:
            raise ValueError("La taille de fenêtre doit être impaire")

        pad = window_size // 2
        height, width = self.image.shape()
        result = np.zeros_like(self.image.data)

        # Padding de l'image
        padded = np.pad(self.image.data, pad, mode='reflect')

        # Traiter chaque pixel
        for y in range(height):
            for x in range(width):
                # Extraire la fenêtre locale
                window = padded[y:y + window_size, x:x + window_size]

                # Calculer l'histogramme local
                hist, _ = np.histogram(window, bins=int(self.image.max_value) + 1,
                                       range=(0, self.image.max_value))
                local_hist = Histogram(hist, int(self.image.max_value) + 1)

                # Égaliser localement
                lut = local_hist.equalize_lut()
                result[y, x] = lut[int(self.image.data[y, x])]

        return Image(result, self.image.max_value)

    def add(self, other, clip: bool = True) -> 'Image':
        """Addition de deux images."""

        if isinstance(other, Image):
            if self.image.shape() != other.shape():
                raise ValueError("Les images doivent avoir la même taille")
            result = self.image.data + other.data
        else:
            result = self.image.data + other

        if clip:
            result = np.clip(result, 0, self.image.max_value)

        return Image(result, self.image.max_value)

    def subtract(self, other, clip: bool = True) -> 'Image':
        """Soustraction de deux images."""

        if isinstance(other, Image):
            if self.image.shape() != other.shape():
                raise ValueError("Les images doivent avoir la même taille")
            result = self.image.data - other.data
        else:
            result = self.image.data - other

        if clip:
            result = np.clip(result, 0, self.image.max_value)

        return Image(result, self.image.max_value)

    def multiply(self, ratio: float) -> 'Image':
        """Multiplication par un ratio."""

        result = self.image.data * ratio
        result = np.clip(result, 0, self.image.max_value)

        return Image(result, self.image.max_value)

    def logical_and(self, other) -> 'Image':
        """Opération ET logique (pour images binaires)."""

        if not isinstance(other, Image):
            raise TypeError("other doit être une Image")
        if self.image.shape() != other.shape():
            raise ValueError("Les images doivent avoir la même taille")

        result = np.logical_and(self.image.data > 0, other.data > 0).astype(np.float64)
        result *= self.image.max_value

        return Image(result, self.image.max_value)

    def logical_or(self, other) -> 'Image':
        """Opération OU logique (pour images binaires)."""

        if not isinstance(other, Image):
            raise TypeError("other doit être une Image")
        if self.image.shape() != other.shape():
            raise ValueError("Les images doivent avoir la même taille")

        result = np.logical_or(self.image.data > 0, other.data > 0).astype(np.float64)
        result *= self.image.max_value

        return Image(result, self.image.max_value)
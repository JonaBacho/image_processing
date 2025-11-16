"""Module processing.interpolation - Méthodes d'interpolation et changement d'échelle."""
from core.image import Image
import numpy as np
from typing import Tuple


class InterpolationProcessor:
    """Processeur pour les méthodes d'interpolation et de redimensionnement."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def nearest_neighbor(self, new_height: int, new_width: int) -> 'Image':
        """
        Interpolation du plus proche voisin (copie directe).

        Args:
            new_height: Nouvelle hauteur
            new_width: Nouvelle largeur
        """
        old_height, old_width = self.image.shape()

        # Ratios de redimensionnement
        row_ratio = old_height / new_height
        col_ratio = old_width / new_width

        # Créer la nouvelle image
        result = np.zeros((new_height, new_width), dtype=np.float64)

        for i in range(new_height):
            for j in range(new_width):
                # Trouver le pixel source le plus proche
                src_i = int(i * row_ratio)
                src_j = int(j * col_ratio)

                # Borner aux dimensions de l'image source
                src_i = min(src_i, old_height - 1)
                src_j = min(src_j, old_width - 1)

                result[i, j] = self.image.data[src_i, src_j]

        return Image(result, self.image.max_value)

    def bilinear(self, new_height: int, new_width: int) -> 'Image':
        """
        Interpolation bilinéaire (utilise 4 pixels voisins).

        Args:
            new_height: Nouvelle hauteur
            new_width: Nouvelle largeur
        """
        old_height, old_width = self.image.shape()

        # Ratios de redimensionnement
        row_ratio = (old_height - 1) / max(new_height - 1, 1)
        col_ratio = (old_width - 1) / max(new_width - 1, 1)

        result = np.zeros((new_height, new_width), dtype=np.float64)

        for i in range(new_height):
            for j in range(new_width):
                # Position dans l'image source
                src_i = i * row_ratio
                src_j = j * col_ratio

                # Coordonnées des 4 pixels voisins
                i1 = int(np.floor(src_i))
                i2 = min(i1 + 1, old_height - 1)
                j1 = int(np.floor(src_j))
                j2 = min(j1 + 1, old_width - 1)

                # Poids pour l'interpolation
                di = src_i - i1
                dj = src_j - j1

                # Interpolation bilinéaire
                # f(x,y) = (1-dx)(1-dy)f(x1,y1) + dx(1-dy)f(x2,y1) + (1-dx)dy*f(x1,y2) + dx*dy*f(x2,y2)
                v1 = self.image.data[i1, j1]
                v2 = self.image.data[i2, j1]
                v3 = self.image.data[i1, j2]
                v4 = self.image.data[i2, j2]

                result[i, j] = (1 - di) * (1 - dj) * v1 + \
                               di * (1 - dj) * v2 + \
                               (1 - di) * dj * v3 + \
                               di * dj * v4

        return Image(result, self.image.max_value)

    def bicubic(self, new_height: int, new_width: int) -> 'Image':
        """
        Interpolation bicubique (utilise 16 pixels voisins).

        Args:
            new_height: Nouvelle hauteur
            new_width: Nouvelle largeur
        """

        old_height, old_width = self.image.shape()

        # Ratios de redimensionnement
        row_ratio = (old_height - 1) / max(new_height - 1, 1)
        col_ratio = (old_width - 1) / max(new_width - 1, 1)

        result = np.zeros((new_height, new_width), dtype=np.float64)

        # Fonction de poids cubique
        def cubic_weight(x):
            """Fonction d'interpolation cubique."""
            x = abs(x)
            if x <= 1:
                return 1.5 * x ** 3 - 2.5 * x ** 2 + 1
            elif x < 2:
                return -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2
            else:
                return 0

        for i in range(new_height):
            for j in range(new_width):
                # Position dans l'image source
                src_i = i * row_ratio
                src_j = j * col_ratio

                # Centre de la grille 4x4
                center_i = int(np.floor(src_i))
                center_j = int(np.floor(src_j))

                # Interpolation bicubique sur 16 pixels
                value = 0.0
                weight_sum = 0.0

                for m in range(-1, 3):
                    for n in range(-1, 3):
                        # Coordonnées du pixel source
                        pi = center_i + m
                        pj = center_j + n

                        # Vérifier les limites
                        if 0 <= pi < old_height and 0 <= pj < old_width:
                            # Calculer les poids
                            wi = cubic_weight(src_i - pi)
                            wj = cubic_weight(src_j - pj)
                            weight = wi * wj

                            value += weight * self.image.data[pi, pj]
                            weight_sum += weight

                # Normaliser si nécessaire
                if weight_sum > 0:
                    result[i, j] = value / weight_sum
                else:
                    result[i, j] = self.image.data[center_i, center_j]

        return Image(np.clip(result, 0, self.image.max_value), self.image.max_value)

    def scale(self, factor: float, method: str = 'bilinear') -> 'Image':
        """
        Redimensionne l'image par un facteur.

        Args:
            factor: Facteur de redimensionnement (ex: 2.0 pour doubler)
            method: Méthode d'interpolation ('nearest', 'bilinear', 'bicubic')
        """
        new_height = int(self.image.height * factor)
        new_width = int(self.image.width * factor)

        if method == 'nearest':
            return self.nearest_neighbor(new_height, new_width)
        elif method == 'bilinear':
            return self.bilinear(new_height, new_width)
        elif method == 'bicubic':
            return self.bicubic(new_height, new_width)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
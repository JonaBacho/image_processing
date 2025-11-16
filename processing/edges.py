"""Module processing.edges - Détection de contours."""

import numpy as np
from typing import Tuple, Optional
from core.image import Image
from core.kernel import Kernel
from processing.spatial import SpatialProcessor


class EdgeDetector:
    """Processeur pour la détection de contours."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def roberts(self) -> 'Image':
        """Détection de contours avec l'opérateur de Roberts."""

        # Appliquer les deux filtres de Roberts
        processor = SpatialProcessor(self.image)
        gx = processor.convolve(Kernel.roberts_x())
        gy = processor.convolve(Kernel.roberts_y())

        # Magnitude du gradient
        magnitude = np.sqrt(gx.data ** 2 + gy.data ** 2)

        return Image(np.clip(magnitude, 0, self.image.max_value), self.image.max_value)

    def prewitt(self) -> 'Image':
        """Détection de contours avec l'opérateur de Prewitt."""

        processor = SpatialProcessor(self.image)
        gx = processor.convolve(Kernel.prewitt_x())
        gy = processor.convolve(Kernel.prewitt_y())

        magnitude = np.sqrt(gx.data ** 2 + gy.data ** 2)

        return Image(np.clip(magnitude, 0, self.image.max_value), self.image.max_value)

    def sobel(self) -> 'Image':
        """Détection de contours avec l'opérateur de Sobel."""

        processor = SpatialProcessor(self.image)
        gx = processor.convolve(Kernel.sobel_x())
        gy = processor.convolve(Kernel.sobel_y())

        magnitude = np.sqrt(gx.data ** 2 + gy.data ** 2)

        return Image(np.clip(magnitude, 0, self.image.max_value), self.image.max_value)

    def gradient_magnitude_angle(self) -> Tuple['Image', np.ndarray]:
        """
        Calcule la magnitude et l'angle du gradient (avec Sobel).

        Returns:
            Tuple (magnitude_image, angle_array en radians)
        """
        processor = SpatialProcessor(self.image)
        gx = processor.convolve(Kernel.sobel_x())
        gy = processor.convolve(Kernel.sobel_y())

        magnitude = np.sqrt(gx.data ** 2 + gy.data ** 2)
        angle = np.arctan2(gy.data, gx.data)

        return Image(np.clip(magnitude, 0, self.image.max_value), self.image.max_value), angle

    def laplacian(self, variant: str = '4-connected') -> 'Image':
        """
        Détection de contours avec le Laplacien (2ème dérivée).

        Args:
            variant: Type de connectivité ('4-connected' ou '8-connected')
        """
        processor = SpatialProcessor(self.image)
        kernel = Kernel.laplacian(variant)
        result = processor.convolve(kernel)

        return result

    def zero_crossing(self, laplacian_image) -> 'Image':
        """
        Détecte les passages par zéro dans une image Laplacienne.

        Args:
            laplacian_image: Image résultant d'un filtre Laplacien
        """

        height, width = laplacian_image.shape()
        result = np.zeros((height, width), dtype=np.float64)

        # Détecter les passages par zéro
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Vérifier le changement de signe avec les voisins
                current = laplacian_image.data[i, j]
                neighbors = [
                    laplacian_image.data[i - 1, j],
                    laplacian_image.data[i + 1, j],
                    laplacian_image.data[i, j - 1],
                    laplacian_image.data[i, j + 1]
                ]

                # Si changement de signe détecté
                for neighbor in neighbors:
                    if current * neighbor < 0:
                        result[i, j] = self.image.max_value
                        break

        return Image(result, self.image.max_value)

    def hough_lines(self, threshold: int = 100, theta_res: int = 180, rho_res: int = 200) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Transformée de Hough pour la détection de lignes.

        Args:
            threshold: Seuil de votes pour considérer une ligne
            theta_res: Résolution angulaire (nombre de bins pour theta)
            rho_res: Résolution en distance (nombre de bins pour rho)

        Returns:
            Tuple (accumulator, thetas, rhos)
        """
        # Obtenir les contours (image binaire)
        edges = self.sobel()

        # Binariser
        binary = (edges.data > edges.data.mean()).astype(np.uint8)

        height, width = edges.shape()

        # Plage des paramètres
        diag_len = int(np.sqrt(height ** 2 + width ** 2))
        thetas = np.linspace(-np.pi / 2, np.pi / 2, theta_res)
        rhos = np.linspace(-diag_len, diag_len, rho_res)

        # Accumulateur
        accumulator = np.zeros((rho_res, theta_res), dtype=np.int32)

        # Indices des pixels de contour
        y_idxs, x_idxs = np.nonzero(binary)

        # Vote pour chaque pixel de contour
        for i in range(len(y_idxs)):
            y = y_idxs[i]
            x = x_idxs[i]

            for t_idx, theta in enumerate(thetas):
                # Calcul de rho = x*cos(theta) + y*sin(theta)
                rho = x * np.cos(theta) + y * np.sin(theta)

                # Trouver l'index de rho le plus proche
                rho_idx = np.argmin(np.abs(rhos - rho))

                # Voter
                accumulator[rho_idx, t_idx] += 1

        # Détecter les lignes au-dessus du seuil
        lines = []
        for rho_idx in range(rho_res):
            for theta_idx in range(theta_res):
                if accumulator[rho_idx, theta_idx] > threshold:
                    lines.append((rhos[rho_idx], thetas[theta_idx]))

        return accumulator, thetas, rhos, lines
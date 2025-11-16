"""Module processing.segmentation - Algorithmes de segmentation."""
from core.image import Image
from scipy import ndimage
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class Segmentation:
    """Processeur pour les algorithmes de segmentation."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def threshold(self, threshold_value: float) -> 'Image':
        """
        Seuillage global simple.

        Args:
            threshold_value: Valeur du seuil
        """

        result = np.where(self.image.data >= threshold_value, self.image.max_value, 0)
        return Image(result, self.image.max_value)

    def multi_threshold(self, thresholds: List[float]) -> 'Image':
        """
        Seuillage multi-niveaux.

        Args:
            thresholds: Liste des seuils triés
        """
        thresholds = sorted(thresholds)
        result = np.zeros_like(self.image.data)

        for i, thresh in enumerate(thresholds):
            level = (i + 1) * (self.image.max_value / (len(thresholds) + 1))
            mask = self.image.data >= thresh
            result[mask] = level

        return Image(result, self.image.max_value)

    def otsu_threshold(self) -> Tuple['Image', float]:
        """
        Algorithme d'Otsu pour le seuillage automatique.

        Returns:
            Tuple (image_seuillée, seuil_optimal)
        """
        # Calculer l'histogramme
        hist = self.image.get_histogram()
        normalized_hist = hist.normalize()

        total_pixels = self.image.width * self.image.height

        # Initialisation
        best_threshold = 0
        best_variance = 0

        # Parcourir tous les seuils possibles
        for t in range(1, hist.bins):
            # Classe 0: pixels [0, t-1]
            w0 = np.sum(normalized_hist.values[:t])
            if w0 == 0:
                continue

            # Classe 1: pixels [t, max]
            w1 = np.sum(normalized_hist.values[t:])
            if w1 == 0:
                continue

            # Moyennes des classes
            levels = np.arange(hist.bins)
            mu0 = np.sum(levels[:t] * normalized_hist.values[:t]) / w0
            mu1 = np.sum(levels[t:] * normalized_hist.values[t:]) / w1

            # Variance inter-classes
            variance_between = w0 * w1 * (mu0 - mu1) ** 2

            # Mise à jour du meilleur seuil
            if variance_between > best_variance:
                best_variance = variance_between
                best_threshold = t

        # Appliquer le seuillage
        result = self.threshold(best_threshold)

        return result, best_threshold

    def adaptive_threshold(self, block_size: int = 11, C: float = 2) -> 'Image':
        """
        Seuillage adaptatif local.

        Args:
            block_size: Taille des blocs (doit être impair)
            C: Constante soustraite de la moyenne
        """
        if block_size % 2 == 0:
            raise ValueError("block_size doit être impair")

        # Calculer la moyenne locale
        local_mean = ndimage.uniform_filter(self.image.data, size=block_size)

        # Seuillage adaptatif
        result = np.where(self.image.data > local_mean - C, self.image.max_value, 0)

        return Image(result, self.image.max_value)

    def kmeans(self, k: int = 2, max_iter: int = 100) -> 'Image':
        """
        Segmentation par K-means clustering.

        Args:
            k: Nombre de clusters
            max_iter: Nombre maximum d'itérations
        """
        # Aplatir l'image
        pixels = self.image.data.flatten().reshape(-1, 1)

        # Initialiser les centres aléatoirement
        min_val, max_val = pixels.min(), pixels.max()
        centers = np.random.uniform(min_val, max_val, (k, 1))

        for iteration in range(max_iter):
            # Affecter chaque pixel au centre le plus proche
            distances = np.abs(pixels - centers.T)
            labels = np.argmin(distances, axis=1)

            # Mettre à jour les centres
            new_centers = np.array([pixels[labels == i].mean() if np.any(labels == i) else centers[i]
                                    for i in range(k)])

            # Vérifier la convergence
            if np.allclose(centers, new_centers.reshape(-1, 1)):
                break

            centers = new_centers.reshape(-1, 1)

        # Créer l'image segmentée
        result = centers[labels].reshape(self.image.shape())

        return Image(result, self.image.max_value)

    def region_growing(self, seeds: List[Tuple[int, int]], threshold: float = 10) -> 'Image':
        """
        Segmentation par croissance de régions.

        Args:
            seeds: Liste de points de départ (y, x)
            threshold: Seuil de similarité
        """

        height, width = self.image.shape()
        segmented = np.zeros((height, width), dtype=np.int32)
        visited = np.zeros((height, width), dtype=bool)

        for region_id, seed in enumerate(seeds, start=1):
            if not (0 <= seed[0] < height and 0 <= seed[1] < width):
                continue

            # File pour le parcours BFS
            queue = deque([seed])
            visited[seed] = True
            region_mean = self.image.data[seed]
            region_pixels = [self.image.data[seed]]

            while queue:
                y, x = queue.popleft()
                segmented[y, x] = region_id

                # Voisins 4-connectés
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx

                    if (0 <= ny < height and 0 <= nx < width and
                            not visited[ny, nx]):

                        pixel_value = self.image.data[ny, nx]

                        # Critère d'homogénéité
                        if abs(pixel_value - region_mean) <= threshold:
                            queue.append((ny, nx))
                            visited[ny, nx] = True
                            region_pixels.append(pixel_value)
                            # Mettre à jour la moyenne
                            region_mean = np.mean(region_pixels)

        # Normaliser les régions
        result = (segmented / segmented.max() * self.image.max_value) if segmented.max() > 0 else segmented

        return Image(result, self.image.max_value)

    def split_and_merge(self, min_size: int = 4, threshold: float = 10) -> 'Image':
        """
        Segmentation par division-fusion (Split-and-Merge).

        Args:
            min_size: Taille minimale d'un bloc
            threshold: Seuil d'homogénéité (écart-type)
        """

        height, width = self.image.shape()
        result = np.zeros((height, width), dtype=np.float64)

        def is_homogeneous(block):
            """Vérifie si un bloc est homogène."""
            return np.std(block) <= threshold

        def split(y, x, h, w, label):
            """Divise récursivement les blocs non-homogènes."""
            block = self.image.data[y:y + h, x:x + w]

            if h <= min_size or w <= min_size or is_homogeneous(block):
                result[y:y + h, x:x + w] = np.mean(block)
                return

            # Diviser en 4 quadrants
            h_half, w_half = h // 2, w // 2
            split(y, x, h_half, w_half, label * 4 + 0)
            split(y, x + w_half, h_half, w - w_half, label * 4 + 1)
            split(y + h_half, x, h - h_half, w_half, label * 4 + 2)
            split(y + h_half, x + w_half, h - h_half, w - w_half, label * 4 + 3)

        split(0, 0, height, width, 1)

        return Image(result, self.image.max_value)
"""Module processing.morphology - Morphologie mathématique et traitement d'images binaires."""
from core.image import Image
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from scipy.ndimage import morphology


class MorphologyProcessor:
    """Processeur pour la morphologie mathématique."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def _create_structuring_element(self, size: int = 3, shape: str = 'square') -> np.ndarray:
        """
        Crée un élément structurant.

        Args:
            size: Taille de l'élément structurant
            shape: Forme ('square', 'cross', 'disk')
        """
        if shape == 'square':
            return np.ones((size, size), dtype=np.uint8)
        elif shape == 'cross':
            element = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            element[center, :] = 1
            element[:, center] = 1
            return element
        elif shape == 'disk':
            y, x = np.ogrid[-size // 2:size // 2 + 1, -size // 2:size // 2 + 1]
            element = (x ** 2 + y ** 2 <= (size // 2) ** 2).astype(np.uint8)
            return element
        else:
            raise ValueError(f"Forme inconnue: {shape}")

    def erosion(self, size: int = 3, shape: str = 'square') -> 'Image':
        """
        Érosion morphologique.
        Rétrécit les objets blancs.

        Args:
            size: Taille de l'élément structurant
            shape: Forme de l'élément structurant
        """
        # Binariser si nécessaire
        binary = (self.image.data > self.image.max_value / 2).astype(np.uint8)

        # Élément structurant
        element = self._create_structuring_element(size, shape)

        # Érosion
        eroded = ndimage.binary_erosion(binary, structure=element).astype(np.float64)
        eroded *= self.image.max_value

        return Image(eroded, self.image.max_value)

    def dilation(self, size: int = 3, shape: str = 'square') -> 'Image':
        """
        Dilatation morphologique.
        Grossit les objets blancs.

        Args:
            size: Taille de l'élément structurant
            shape: Forme de l'élément structurant
        """
        # Binariser si nécessaire
        binary = (self.image.data > self.image.max_value / 2).astype(np.uint8)

        # Élément structurant
        element = self._create_structuring_element(size, shape)

        # Dilatation
        dilated = ndimage.binary_dilation(binary, structure=element).astype(np.float64)
        dilated *= self.image.max_value

        return Image(dilated, self.image.max_value)

    def opening(self, size: int = 3, shape: str = 'square') -> 'Image':
        """
        Ouverture morphologique (érosion puis dilatation).
        Élimine le bruit et lisse les contours.

        Args:
            size: Taille de l'élément structurant
            shape: Forme de l'élément structurant
        """
        # Érosion suivie de dilatation
        eroded = self.erosion(size, shape)
        morph = MorphologyProcessor(eroded)
        return morph.dilation(size, shape)

    def closing(self, size: int = 3, shape: str = 'square') -> 'Image':
        """
        Fermeture morphologique (dilatation puis érosion).
        Comble les petits trous et ferme les contours.

        Args:
            size: Taille de l'élément structurant
            shape: Forme de l'élément structurant
        """
        # Dilatation suivie d'érosion
        dilated = self.dilation(size, shape)
        morph = MorphologyProcessor(dilated)
        return morph.erosion(size, shape)

    def connected_components(self) -> Tuple['Image', int]:
        """
        Étiquetage des composantes connexes.

        Returns:
            Tuple (image_étiquetée, nombre_de_composantes)
        """
        # Binariser
        binary = (self.image.data > self.image.max_value / 2).astype(np.uint8)

        # Étiquetage
        labeled, num_features = ndimage.label(binary)

        # Normaliser pour visualisation
        if num_features > 0:
            labeled = (labeled / num_features * self.image.max_value).astype(np.float64)

        return Image(labeled, self.image.max_value), num_features

    def hysteresis_threshold(self, low_threshold: float, high_threshold: float) -> 'Image':
        """
        Seuillage par hystérésis (utilisé dans Canny).

        Args:
            low_threshold: Seuil bas
            high_threshold: Seuil haut
        """
        # Pixels forts (au-dessus du seuil haut)
        strong = self.image.data >= high_threshold

        # Pixels faibles (entre les deux seuils)
        weak = (self.image.data >= low_threshold) & (self.image.data < high_threshold)

        # Étiquetage des pixels forts
        labeled_strong, _ = ndimage.label(strong)

        # Dilatation des pixels forts pour capturer les pixels faibles connectés
        dilated_strong = ndimage.binary_dilation(strong)

        # Garder seulement les pixels faibles connectés aux pixels forts
        result = strong.astype(np.float64)
        result[weak & dilated_strong] = 1
        result *= self.image.max_value

        return Image(result, self.image.max_value)

    def skeleton(self) -> 'Image':
        """
        Squelettisation (réduction à une ligne centrale).
        """

        # Binariser
        binary = (self.image.data > self.image.max_value / 2).astype(bool)

        # Squelettisation
        skel = morphology.binary_erosion(binary)
        skeleton = np.zeros_like(binary, dtype=bool)

        # Algorithme de squelettisation itératif
        while np.any(skel):
            opened = morphology.binary_opening(skel)
            skeleton |= (skel & ~opened)
            skel = morphology.binary_erosion(skel)

        result = skeleton.astype(np.float64) * self.image.max_value

        return Image(result, self.image.max_value)

    def fill_holes(self) -> 'Image':
        """
        Remplit les trous dans les objets binaires.
        """

        # Binariser
        binary = (self.image.data > self.image.max_value / 2).astype(bool)

        # Remplir les trous
        filled = ndimage.binary_fill_holes(binary).astype(np.float64)
        filled *= self.image.max_value

        return Image(filled, self.image.max_value)

    def boundary_extraction(self, size: int = 3) -> 'Image':
        """
        Extraction de contours par différence morphologique.

        Args:
            size: Taille de l'élément structurant
        """
        # Binariser
        binary_img = self.image.data > self.image.max_value / 2

        # Érosion
        eroded = self.erosion(size)

        # Contour = Image originale - Image érodée
        boundary = binary_img.astype(np.float64) - (eroded.data > 0).astype(np.float64)
        boundary = np.clip(boundary, 0, 1) * self.image.max_value

        return Image(boundary, self.image.max_value)
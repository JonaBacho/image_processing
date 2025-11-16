"""Module processing.frequency - Traitement dans le domaine fréquentiel."""
from core.image import Image
import numpy as np
from typing import Tuple, Optional



class FrequencyProcessor:
    """Processeur pour le traitement dans le domaine fréquentiel (FFT)."""

    def __init__(self, image):
        if not isinstance(image, Image):
            raise TypeError("L'image doit être une instance de la classe Image")
        self.image = image

    def fft(self) -> np.ndarray:
        """
        Calcule la transformée de Fourier rapide (FFT) 2D.

        Returns:
            Spectre de Fourier (complexe) centré
        """
        # FFT 2D
        f_transform = np.fft.fft2(self.image.data)
        # Centrer le spectre (basses fréquences au centre)
        f_shifted = np.fft.fftshift(f_transform)
        return f_shifted

    def ifft(self, spectrum: np.ndarray) -> 'Image':
        """
        Calcule la transformée de Fourier inverse (IFFT).

        Args:
            spectrum: Spectre de Fourier centré

        Returns:
            Image reconstruite
        """

        # Décentrer le spectre
        f_ishifted = np.fft.ifftshift(spectrum)
        # IFFT 2D
        img_back = np.fft.ifft2(f_ishifted)
        # Prendre la partie réelle
        result = np.real(img_back)

        return Image(np.clip(result, 0, self.image.max_value), self.image.max_value)

    def get_magnitude_spectrum(self) -> 'Image':
        """
        Calcule le spectre de magnitude (pour visualisation).

        Returns:
            Image du spectre de magnitude (échelle logarithmique)
        """
        spectrum = self.fft()
        magnitude = np.abs(spectrum)

        # Échelle logarithmique pour meilleure visualisation
        magnitude_log = 20 * np.log(magnitude + 1)

        # Normaliser à [0, 255]
        magnitude_normalized = (magnitude_log - magnitude_log.min())
        if magnitude_log.max() - magnitude_log.min() > 0:
            magnitude_normalized = magnitude_normalized / (magnitude_log.max() - magnitude_log.min()) * 255

        return Image(magnitude_normalized, 255)

    def lowpass_filter(self, cutoff_freq: int) -> 'Image':
        """
        Filtre passe-bas dans le domaine fréquentiel.
        Met à zéro les hautes fréquences (loin du centre).

        Args:
            cutoff_freq: Rayon de coupure (en pixels depuis le centre)
        """
        # FFT
        spectrum = self.fft()

        # Créer le masque passe-bas circulaire
        rows, cols = self.image.shape()
        center_row, center_col = rows // 2, cols // 2

        mask = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                if distance <= cutoff_freq:
                    mask[i, j] = 1

        # Appliquer le masque
        filtered_spectrum = spectrum * mask

        # IFFT
        return self.ifft(filtered_spectrum)

    def highpass_filter(self, cutoff_freq: int) -> 'Image':
        """
        Filtre passe-haut dans le domaine fréquentiel.
        Met à zéro les basses fréquences (au centre).

        Args:
            cutoff_freq: Rayon de coupure (en pixels depuis le centre)
        """

        # FFT
        spectrum = self.fft()

        # Créer le masque passe-haut circulaire
        rows, cols = self.image.shape()
        center_row, center_col = rows // 2, cols // 2

        mask = np.ones((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                if distance <= cutoff_freq:
                    mask[i, j] = 0

        # Appliquer le masque
        filtered_spectrum = spectrum * mask

        # IFFT
        return self.ifft(filtered_spectrum)

    def bandpass_filter(self, low_cutoff: int, high_cutoff: int) -> 'Image':
        """
        Filtre passe-bande dans le domaine fréquentiel.

        Args:
            low_cutoff: Rayon de coupure inférieur
            high_cutoff: Rayon de coupure supérieur
        """
        # FFT
        spectrum = self.fft()

        # Créer le masque passe-bande circulaire
        rows, cols = self.image.shape()
        center_row, center_col = rows // 2, cols // 2

        mask = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                if low_cutoff <= distance <= high_cutoff:
                    mask[i, j] = 1

        # Appliquer le masque
        filtered_spectrum = spectrum * mask

        # IFFT
        return self.ifft(filtered_spectrum)
"""Module utils.io_handler - Lecture et écriture de fichiers image."""
import numpy as np
from pathlib import Path


class IOHandler:
    """Gestionnaire pour lire et écrire des fichiers PGM/PPM."""

    @staticmethod
    def load_image(filepath: str):
        """Charge une image depuis un fichier PGM ou PPM."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {filepath}")

        with open(filepath, 'rb') as f:
            # Lire l'en-tête
            magic = f.readline().decode('ascii').strip()

            # Ignorer les commentaires
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()

            # Lire dimensions
            width, height = map(int, line.decode('ascii').split())

            # Lire valeur max
            max_val = int(f.readline().decode('ascii').strip())

            # Lire les données
            if magic == 'P2':  # PGM ASCII
                data = np.fromfile(f, dtype=np.int32, count=width * height, sep=' ')
            elif magic == 'P5':  # PGM binaire
                data = np.fromfile(f, dtype=np.uint8, count=width * height)
            elif magic == 'P3':  # PPM ASCII
                data = np.fromfile(f, dtype=np.int32, count=width * height * 3, sep=' ')
                data = data.reshape((height, width, 3))
                # Convertir en niveaux de gris
                data = np.mean(data, axis=2)
                data = data.flatten()
            elif magic == 'P6':  # PPM binaire
                data = np.fromfile(f, dtype=np.uint8, count=width * height * 3)
                data = data.reshape((height, width, 3))
                # Convertir en niveaux de gris
                data = np.mean(data, axis=2)
                data = data.flatten()
            else:
                raise ValueError(f"Format non supporté: {magic}")

            # Reshape en 2D
            data = data.reshape((height, width))

            return data, max_val

    @staticmethod
    def save_image(image, filepath: str, format: str = 'pgm'):
        """Sauvegarde une image dans un fichier PGM ou PPM."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convertir en entiers
        data = np.clip(image.data, 0, image.max_value).astype(np.uint8)

        with open(filepath, 'wb') as f:
            if format.lower() == 'pgm':
                # En-tête PGM binaire
                f.write(b'P5\n')
                f.write(f'{image.width} {image.height}\n'.encode('ascii'))
                f.write(f'{image.max_value}\n'.encode('ascii'))
                # Données
                data.tofile(f)
            elif format.lower() == 'ppm':
                # Convertir en RGB (3 canaux identiques pour le gris)
                rgb_data = np.stack([data, data, data], axis=-1)
                # En-tête PPM binaire
                f.write(b'P6\n')
                f.write(f'{image.width} {image.height}\n'.encode('ascii'))
                f.write(f'{image.max_value}\n'.encode('ascii'))
                # Données
                rgb_data.tofile(f)
            else:
                raise ValueError(f"Format non supporté: {format}")
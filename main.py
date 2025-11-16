#!/usr/bin/env python3
"""
Main CLI - Interface en ligne de commande pour la bibliothèque de traitement d'images.

Usage:
    python main.py --input image.pgm --output result.pgm --operation sobel
    python main.py -i image.pgm -o result.pgm -op histogram_equalization
    python main.py -i image.pgm -o result.pgm -op gaussian --size 5 --sigma 1.5
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image import Image
from processing.pointwise import PointwiseProcessor
from processing.interpolation import InterpolationProcessor
from processing.spatial import SpatialProcessor
from processing.frequency import FrequencyProcessor
from processing.edges import EdgeDetector
from processing.segmentation import Segmentation
from processing.morphology import MorphologyProcessor


def list_operations():
    """Liste toutes les opérations disponibles."""
    operations = {
        "Traitements Ponctuels": [
            "linear_transform",
            "gamma_correction",
            "histogram_equalization",
            "local_histogram_equalization",
        ],
        "Interpolation": [
            "resize_nearest",
            "resize_bilinear",
            "resize_bicubic",
        ],
        "Filtrage Spatial": [
            "mean_filter",
            "gaussian_filter",
            "median_filter",
        ],
        "Domaine Fréquentiel": [
            "lowpass_filter",
            "highpass_filter",
        ],
        "Détection de Contours": [
            "roberts",
            "prewitt",
            "sobel",
            "laplacian",
        ],
        "Segmentation": [
            "threshold",
            "otsu",
            "kmeans",
        ],
        "Morphologie": [
            "erosion",
            "dilation",
            "opening",
            "closing",
            "connected_components",
        ]
    }

    print("\n=== Opérations Disponibles ===\n")
    for category, ops in operations.items():
        print(f"{category}:")
        for op in ops:
            print(f"  - {op}")
        print()


def apply_operation(image, operation, args):
    """Applique l'opération spécifiée sur l'image."""

    # === TRAITEMENTS PONCTUELS ===
    if operation == "linear_transform":
        processor = PointwiseProcessor(image)
        return processor.linear_transform(
            s_min=args.min_val if hasattr(args, 'min_val') else None,
            s_max=args.max_val if hasattr(args, 'max_val') else None
        )

    elif operation == "gamma_correction":
        processor = PointwiseProcessor(image)
        gamma = args.gamma if hasattr(args, 'gamma') else 1.0
        return processor.gamma_correction(gamma=gamma)

    elif operation == "histogram_equalization":
        processor = PointwiseProcessor(image)
        return processor.histogram_equalization()

    elif operation == "local_histogram_equalization":
        processor = PointwiseProcessor(image)
        window_size = args.window_size if hasattr(args, 'window_size') else 7
        return processor.local_histogram_equalization(window_size=window_size)

    # === INTERPOLATION ===
    elif operation == "resize_nearest":
        processor = InterpolationProcessor(image)
        return processor.nearest_neighbor(args.height, args.width)

    elif operation == "resize_bilinear":
        processor = InterpolationProcessor(image)
        return processor.bilinear(args.height, args.width)

    elif operation == "resize_bicubic":
        processor = InterpolationProcessor(image)
        return processor.bicubic(args.height, args.width)

    # === FILTRAGE SPATIAL ===
    elif operation == "mean_filter":
        processor = SpatialProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        return processor.mean_filter(size=size)

    elif operation == "gaussian_filter":
        processor = SpatialProcessor(image)
        size = args.size if hasattr(args, 'size') else 5
        sigma = args.sigma if hasattr(args, 'sigma') else 1.0
        return processor.gaussian_filter(size=size, sigma=sigma)

    elif operation == "median_filter":
        processor = SpatialProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        return processor.median_filter(size=size)

    # === DOMAINE FRÉQUENTIEL ===
    elif operation == "lowpass_filter":
        processor = FrequencyProcessor(image)
        cutoff = args.cutoff if hasattr(args, 'cutoff') else 30
        return processor.lowpass_filter(cutoff_freq=cutoff)

    elif operation == "highpass_filter":
        processor = FrequencyProcessor(image)
        cutoff = args.cutoff if hasattr(args, 'cutoff') else 10
        return processor.highpass_filter(cutoff_freq=cutoff)

    # === DÉTECTION DE CONTOURS ===
    elif operation == "roberts":
        detector = EdgeDetector(image)
        return detector.roberts()

    elif operation == "prewitt":
        detector = EdgeDetector(image)
        return detector.prewitt()

    elif operation == "sobel":
        detector = EdgeDetector(image)
        return detector.sobel()

    elif operation == "laplacian":
        detector = EdgeDetector(image)
        variant = args.variant if hasattr(args, 'variant') else '4-connected'
        return detector.laplacian(variant=variant)

    # === SEGMENTATION ===
    elif operation == "threshold":
        seg = Segmentation(image)
        threshold = args.threshold if hasattr(args, 'threshold') else 128
        return seg.threshold(threshold_value=threshold)

    elif operation == "otsu":
        seg = Segmentation(image)
        result, threshold = seg.otsu_threshold()
        print(f"Seuil optimal trouvé: {threshold:.2f}")
        return result

    elif operation == "kmeans":
        seg = Segmentation(image)
        k = args.k if hasattr(args, 'k') else 2
        return seg.kmeans(k=k)

    # === MORPHOLOGIE ===
    elif operation == "erosion":
        morph = MorphologyProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        shape = args.shape if hasattr(args, 'shape') else 'square'
        return morph.erosion(size=size, shape=shape)

    elif operation == "dilation":
        morph = MorphologyProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        shape = args.shape if hasattr(args, 'shape') else 'square'
        return morph.dilation(size=size, shape=shape)

    elif operation == "opening":
        morph = MorphologyProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        shape = args.shape if hasattr(args, 'shape') else 'square'
        return morph.opening(size=size, shape=shape)

    elif operation == "closing":
        morph = MorphologyProcessor(image)
        size = args.size if hasattr(args, 'size') else 3
        shape = args.shape if hasattr(args, 'shape') else 'square'
        return morph.closing(size=size, shape=shape)

    elif operation == "connected_components":
        morph = MorphologyProcessor(image)
        result, num = morph.connected_components()
        print(f"Nombre de composantes connexes: {num}")
        return result

    else:
        raise ValueError(f"Opération inconnue: {operation}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Bibliothèque de traitement d\'images - Interface CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Égalisation d'histogramme
  python main.py -i input.pgm -o output.pgm -op histogram_equalization

  # Détection de contours avec Sobel
  python main.py -i input.pgm -o output.pgm -op sobel

  # Filtre gaussien avec paramètres
  python main.py -i input.pgm -o output.pgm -op gaussian_filter --size 5 --sigma 1.5

  # Segmentation Otsu
  python main.py -i input.pgm -o output.pgm -op otsu

  # Redimensionnement
  python main.py -i input.pgm -o output.pgm -op resize_bilinear --height 512 --width 512

  # Lister toutes les opérations
  python main.py --list
        """
    )

    # Arguments principaux
    parser.add_argument('-i', '--input', type=str, help='Chemin de l\'image source')
    parser.add_argument('-o', '--output', type=str, help='Chemin de l\'image de sortie')
    parser.add_argument('-op', '--operation', type=str, help='Opération à effectuer')
    parser.add_argument('--list', action='store_true', help='Liste toutes les opérations disponibles')

    # Arguments optionnels pour les différentes opérations
    parser.add_argument('--size', type=int, help='Taille du filtre ou de l\'élément structurant')
    parser.add_argument('--sigma', type=float, help='Écart-type pour le filtre gaussien')
    parser.add_argument('--gamma', type=float, help='Valeur gamma pour la correction gamma')
    parser.add_argument('--min-val', type=float, help='Valeur minimale pour transformation linéaire')
    parser.add_argument('--max-val', type=float, help='Valeur maximale pour transformation linéaire')
    parser.add_argument('--threshold', type=float, help='Valeur de seuil pour le seuillage')
    parser.add_argument('--k', type=int, help='Nombre de clusters pour K-means')
    parser.add_argument('--cutoff', type=int, help='Fréquence de coupure pour filtres fréquentiels')
    parser.add_argument('--height', type=int, help='Hauteur pour redimensionnement')
    parser.add_argument('--width', type=int, help='Largeur pour redimensionnement')
    parser.add_argument('--window-size', type=int, help='Taille de fenêtre pour égalisation locale')
    parser.add_argument('--shape', type=str, choices=['square', 'cross', 'disk'],
                        help='Forme de l\'élément structurant')
    parser.add_argument('--variant', type=str, choices=['4-connected', '8-connected'],
                        help='Variante pour le laplacien')

    args = parser.parse_args()

    # Lister les opérations
    if args.list:
        list_operations()
        return 0

    # Vérifier les arguments obligatoires
    if not args.input or not args.output or not args.operation:
        parser.print_help()
        print("\n❌ Erreur: Les arguments --input, --output et --operation sont obligatoires")
        return 1

    try:
        # Charger l'image
        print(f"Chargement de l'image: {args.input}")
        image = Image.from_file(args.input)
        print(f"✓ Image chargée: {image.shape()}")

        # Appliquer l'opération
        print(f"⚙Application de l'opération: {args.operation}")
        result = apply_operation(image, args.operation, args)
        print(f"✓ Opération terminée")

        # Sauvegarder le résultat
        print(f"Sauvegarde du résultat: {args.output}")
        result.to_file(args.output)
        print(f"✓ Image sauvegardée avec succès")

        print(f"\n✅ Traitement terminé avec succès!")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Erreur: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
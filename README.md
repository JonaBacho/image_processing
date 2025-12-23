# Projet de Traitement d'Image - Bibliothèque Python
Ce projet est une bibliothèque de traitement d'images développée en Python. Elle permet d'effectuer diverses opérations allant des transformations ponctuelles de base à la segmentation avancée et à la morphologie mathématique, le tout via une interface en ligne de commande (CLI).

## Installation et Configuration
Le projet a été développé pour Python 3.11+. Les dépendances principales sont numpy (calcul matriciel) et scipy (traitements avancés).

1. Prérequis (Ubuntu / Linux)
Ouvrez un terminal et installez Python et le gestionnaire de paquets :

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

2. Prérequis (Windows)
Téléchargez et installez Python 3.11+ depuis python.org.

Assurez-vous de cocher l'option "Add Python to PATH" lors de l'installation.

3. Mise en place de l'environnement (Recommandé)
Il est conseillé d'utiliser un environnement virtuel pour ne pas polluer votre système.

Sous Ubuntu/Linux :
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy
```

Sous Windows :
```bash
python -m venv venv
.\venv\Scripts\activate
pip install numpy scipy
```

# Structure du Projet
Le projet est organisé de manière modulaire :
```text
.
├── core/                # Noyau : classes Image, Histogram et gestion des Kernels
├── processing/          # Algorithmes de traitement
│   ├── pointwise.py     # Transformations ponctuelles (Luminosité, Gamma...)
│   ├── spatial.py       # Filtres spatiaux (Moyenneur, Gaussien...)
│   ├── frequency.py     # Filtres fréquentiels (Passe-bas, Passe-haut)
│   ├── edges.py         # Détection de contours (Sobel, Canny...)
│   ├── interpolation.py # Redimensionnement (Bilinéaire, Bicubique...)
│   ├── segmentation.py  # Otsu, K-means, Seuillage...
│   └── morphology.py    # Érosion, Dilatation, Composantes connexes...
├── utils/               # Utilitaires (Lecture/Écriture PGM, validations)
├── images/              # Images de test (.pgm)
├── tests/               # Tests unitaires
└── main.py              # Point d'entrée CLI
```

## Utilisation
Le fichier main.py permet d'exécuter tous les traitements.

## Lister les opérations disponibles
Pour voir toutes les transformations que vous pouvez appliquer :

```bash
python3 main.py --list
```

## Syntaxe générale
```bash
python3 main.py -i <input_path> -o <output_path> -op <operation> [options]
```

## Exemples concrets (Commandes testées)
1. Égalisation d'histogramme simple :

```bash
python3 main.py -i images/chat.pgm -o images/chat_histogram.pgm -op histogram_equalization
```

2. Égalisation locale d'histogramme (fenêtre de 7x7) :

```bash
python3 main.py -i images/chat.pgm -o images/chat_local_equalize_histogram.pgm -op local_histogram_equalization --window-size 7
```

3. Transformation linéaire (Recadrage de dynamique) :
```bash
python3 main.py -i images/chat.pgm -o images/chat_linear_50_200.pgm -op linear_transform --min-val 50 --max-val 200
```

4. Détection de contours (Sobel) :

```bash
python3 main.py -i images/chat.pgm -o images/contours.pgm -op sobel
```

# Liste des fonctionnalités
- **Traitements Ponctuels** : Transformation linéaire, Correction gamma, Égalisation d'histogramme (globale et locale).

- **Interpolation** : Plus proche voisin, Bilinéaire, Bicubique.

- **Filtrage Spatial** : Filtre moyenneur, gaussien et médian.

- **Domaine Fréquentiel** : Filtres passe-haut et passe-bas.

- **Détection de Contours** : Roberts, Prewitt, Sobel, Laplacien.

- **Segmentation** : Seuillage manuel, méthode d'Otsu, K-means.

- **Morphologie** : Érosion, Dilatation, Ouverture, Fermeture et étiquetage de composantes connexes.

# Notes pour l'enseignant

- Le code respecte les principes de la programmation orientée objet.

- Les images sont principalement gérées au format .pgm (via utils/io_handlers.py).

- Aucune bibliothèque de haut niveau comme OpenCV n'a été utilisée pour les algorithmes de traitement afin de garantir une implémentation "from scratch" via NumPy.

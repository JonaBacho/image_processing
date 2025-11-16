"""Processing module - Algorithmes de traitement d'images."""

from processing.pointwise import PointwiseProcessor
from processing.interpolation import InterpolationProcessor
from processing.spatial import SpatialProcessor
from processing.frequency import FrequencyProcessor
from processing.edges import EdgeDetector
from processing.segmentation import Segmentation
from processing.morphology import MorphologyProcessor

__all__ = [
    'PointwiseProcessor',
    'InterpolationProcessor',
    'SpatialProcessor',
    'FrequencyProcessor',
    'EdgeDetector',
    'Segmentation',
    'MorphologyProcessor'
]
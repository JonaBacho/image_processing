"""Module utils.validators - Validation des données."""

import numpy as np


class Validators:
    """Classe pour valider les paramètres et données."""

    @staticmethod
    def validate_positive_int(value: int, name: str = "valeur") -> int:
        """Valide qu'une valeur est un entier positif."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} doit être un entier positif")
        return value

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, name: str = "valeur") -> float:
        """Valide qu'une valeur est dans un intervalle."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} doit être entre {min_val} et {max_val}")
        return value

    @staticmethod
    def validate_kernel_size(size: int) -> int:
        """Valide qu'une taille de noyau est impaire et positive."""
        if size % 2 == 0 or size < 1:
            raise ValueError("La taille du noyau doit être un entier impair positif")
        return size
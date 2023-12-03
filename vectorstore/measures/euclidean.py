import numpy as np

from vectorstore.measures.measure import AbstractMeasure
from models.vec import Vector


class EuclideanDistance(AbstractMeasure):
    """Euclidean distance measure."""

    def calculate(self, vec1: Vector, vec2: Vector) -> float:
        """Calculate the Euclidean distance between two vectors."""
        return np.linalg.norm((vec1 - vec2).embedding)

    def sort_func(self, x):
        """For Euclidean distance, smaller scores indicate higher similarity. Since the
        default sorting is in ascending order, we can return the score as is."""
        return x

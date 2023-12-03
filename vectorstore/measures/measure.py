from abc import ABC, abstractmethod

from models.vec import Vector


class AbstractMeasure(ABC):
    """Abstract base class for similarity measures."""

    @abstractmethod
    def calculate(self, vec1: Vector, vec2: Vector) -> float:
        """Calculates the similarity between two vectors.

        Args:
            vec1 (Vector): The first vector.
            vec2 (Vector): The second vector.

        Returns:
            float: The similarity score between the two vectors.
        """
        pass

    @abstractmethod
    def sort_func(self, x):
        """Sorting function for the score of the similarity measure, to be able to return
        the most similar vectors. This sorting might differ between similarity measures.

        Args:
            x: The input to be sorted.

        Returns:
            The sorted input.
        """
        pass

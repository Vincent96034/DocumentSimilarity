from typing import Literal

from vectorstore.measures import (
    EuclideanDistance, DotProductDistance, CosineSimilarity)


class MeasureFactory:
    """Factory class for measures."""

    MEASURES = {
        "euclidean": EuclideanDistance(),
        "dot": DotProductDistance(),
        "cosine": CosineSimilarity(),
    }

    @staticmethod
    def get_measure(name: str) -> Literal["AbstractMeasure"]:
        """Get a measure object based on the given name.

        Args:
            name (str): The name of the measure.

        Returns:
            AbstractMeasure: The measure object corresponding to the given name.

        Raises:
            ValueError: If the name does not exist in the dictionary.
        """
        measure = MeasureFactory.MEASURES.get(name)
        if measure is None:
            raise ValueError(f"Measure '{name}' does not exist. Please choose from "
                             f"{list(MeasureFactory.MEASURES.keys())}")
        return measure

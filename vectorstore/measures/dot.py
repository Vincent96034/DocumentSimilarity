from vectorstore.measures.measure import AbstractMeasure
from models.vec import Vector


class DotProductDistance(AbstractMeasure):
    """Dot product distance measure."""

    def calculate(self, vec1: Vector, vec2: Vector) -> float:
        """Calculate the dot product between two vectors."""
        return vec1.dot(vec2)

    def sort_func(self, x):
        """Sort by descending dot product. A positive dot product indicates that the
        vectors are pointing in a generally similar direction. The larger the dot product,
        the more aligned the vectors are."""
        return -x

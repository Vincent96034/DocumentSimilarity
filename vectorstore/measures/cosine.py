from vectorstore.measures.measure import AbstractMeasure
from models.vec import Vector


class CosineSimilarity(AbstractMeasure):
    """Cosine similarity measure."""

    def calculate(self, vec1: Vector, vec2: Vector) -> float:
        """Calculate the cosine similarity between two vectors.
        Following: https://en.wikipedia.org/wiki/Cosine_similarity
        """
        return vec1.dot(vec2) / (vec1.norm() * vec2.norm())

    def sort_func(self, x):
        """Sort by descending cosine similarity. Cosine similarity can take values between
        -1 and 1, where 1 is the most similar and -1 is the least similar."""
        return -x

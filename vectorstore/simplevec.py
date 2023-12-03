import uuid
import random

from vectorstore.vectordb import AbstractVectorDatabase
from models.vec import Vector
from utils.logger_config import logger


class SimpleVectorDatabase(AbstractVectorDatabase):
    """A simple vector database implementation.

    This class represents a simple vector database that stores vectors of a given
    dimension. It provides methods for inserting, updating, deleting, and searching
    vectors in the database.

    Attributes:
        data (dict): A dictionary that stores the vectors in the database.
        vector_dim (int): The dimension of the vectors stored in the database.
    """

    def __init__(self, vector_dim: int):
        self.data = {}
        self.vector_dim = vector_dim

    def _get_vector(self, vec_id: uuid.UUID) -> Vector:
        """Retrieves a vector from the database."""
        vector = self.data.get(vec_id, None)
        if vector is None:
            logger.warning(f"Vector {vec_id} not found in database.")
        return vector

    def _upsert(self, vector: Vector):
        """Inserts or updates a vector in the database."""
        if vector.vector_dim != self.vector_dim:
            raise ValueError((
                f"Vector dimension {vector.vector_dim} does not match database "
                f"dimension {self.vector_dim}."))
        self.data[vector.vec_id] = vector

    def _delete(self, vector: Vector):
        """Deletes a vector from the database."""
        del self.data[vector.vec_id]

    def _sim_search(self, vector, measure, k, **kwargs):
        """Performs a similarity search in the database."""
        similar = {}
        for vec in self.data.values():
            score = measure.calculate(vector, vec, **kwargs)
            similar[vec.vec_id] = {"vector": vec, "score": score}
        similar_vectors = sorted(
            similar.values(), key=lambda x: measure.sort_func(x["score"]))
        return similar_vectors[:k]

    def _get_random_vector(self) -> Vector:
        """Retrieves a random vector from the database."""
        if len(self.data) == 0:
            raise ValueError("Database is empty.")
        random_index = random.randint(0, len(self.data) - 1)
        return list(self.data.values())[random_index]

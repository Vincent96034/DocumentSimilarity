from abc import ABC, abstractmethod
import uuid
from typing import Union, List

from models._types import ArrayLike
from models.vec import Vector
from vectorstore.measures.measure import AbstractMeasure
from vectorstore.measurefactory import MeasureFactory
from utils.logger_config import logger


class AbstractVectorDatabase(ABC):
    """Abstract base class for a vector database.
    This class defines the interface for interacting with a vector database. Subclasses
    must implement the abstract methods.

    Attributes:
        None

    Methods:
        get_vector: Get a vector from the database.
        get_vectors: Get multiple vectors from the database.
        upsert: Update/insert a vector into the database.
        bulk_upsert: Upsert multiple vectors into the database.
        delete: Delete a vector from the database.
        sim_search: Perform a similarity search in the database.
        get_random_vector: Get a random vector from the database.
    """

    @abstractmethod
    def _get_vector(self, vec_id) -> Vector:
        """Get a vector from the database.

        Args:
            vec_id: The ID of the vector to retrieve.

        Returns:
            The vector corresponding to the given ID.
        """
        pass

    @abstractmethod
    def _upsert(self, vector: Vector):
        """Upsert a vector into the database.

        Args:
            vector: The vector to upsert.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _delete(self, vector: Vector):
        """Delete a vector from the database.

        Args:
            vector: The vector to delete.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _sim_search(self, vector: Vector, measure: AbstractMeasure, **kwargs):
        """Perform a similarity search in the database.

        Args:
            vector: The query vector for the similarity search.
            measure: The similarity measure to use.
            k: The number of nearest neighbors to return.
            **kwargs: Additional keyword arguments for the similarity measure.

        Returns:
            A list of k nearest neighbor vectors.
        """
        pass

    @abstractmethod
    def get_random_vector(self) -> Vector:
        """Get a random vector from the database.

        Args:
            None

        Returns:
            A random vector from the database.
        """
        pass

    def get_vector(self, vec_id) -> Vector:
        """Get a vector from the database.

        Args:
            vec_id: The ID of the vector to retrieve.

        Returns:
            The vector corresponding to the given ID.
        """
        if isinstance(vec_id, str):
            vec_id = uuid.UUID(vec_id)
        if not isinstance(vec_id, uuid.UUID):
            raise TypeError(f"Invalid type {type(vec_id)}. Must be UUID.")
        return self._get_vector(vec_id)

    def get_vectors(self, vec_ids) -> list:
        """Get multiple vectors from the database.

        Args:
            vec_ids: A list of IDs of the vectors to retrieve.

        Returns:
            A list of vectors corresponding to the given IDs.
        """
        return [self.get_vector(vec_id) for vec_id in vec_ids]

    def upsert(self, vector: Union[Vector, ArrayLike]):
        """Upsert a vector into the database.

        Args:
            vector: The vector to upsert.

        Returns:
            None
        """
        if isinstance(vector, ArrayLike):
            vector = Vector(embedding=vector)
        if not isinstance(vector, Vector):
            raise TypeError(f"Invalid type {type(vector)}")
        return self._upsert(vector)

    def bulk_upsert(self, vectors: List[Union[Vector, ArrayLike]]):
        """Upsert multiple vectors into the database.

        Args:
            vectors: A list of vectors to upsert.

        Returns:
            None
        """
        if not isinstance(vectors, list):
            raise TypeError(f"Invalid type {type(vectors)}. Must be list.")
        for vector in vectors:
            if isinstance(vector, ArrayLike):
                vector = Vector(embedding=vector)
            if not isinstance(vector, Vector):
                raise TypeError(f"Invalid type {type(vector)}")
        for vector in vectors:
            self._upsert(vector)

    def delete(self, vector_or_id: Union[Vector, uuid.UUID, str]):
        """Delete a vector from the database.

        Args:
            vector_or_id: The vector or ID of the vector to delete.

        Returns:
            None
        """
        if isinstance(vector_or_id, (uuid.UUID, str)):
            vector = self.get_vector(vector_or_id)
        if isinstance(vector_or_id, Vector):
            vector = vector_or_id
        if vector is None:
            logger.warning(f"Vector {vector_or_id} not found in database.")
            return
        if not isinstance(vector, Vector):
            raise TypeError(
                f"Invalid type {type(vector_or_id)}. Must be Vector or vec_id.")
        self._delete(vector)

    def sim_search(self,
                   vector: Union[Vector, ArrayLike],
                   measure: Union[str, AbstractMeasure] = "dot",
                   k: int = 10,
                   **kwargs
                   ) -> List[Vector]:
        """Perform a similarity search in the database.

        Args:
            vector: The query vector for the similarity search.
            measure: The similarity measure to use.
            k: The number of nearest neighbors to return.
            **kwargs: Additional keyword arguments for the similarity measure.

        Returns:
            A list of k nearest neighbor vectors.
        """
        if isinstance(vector, ArrayLike):
            vector = Vector(embedding=vector)
        if not isinstance(vector, Vector):
            raise TypeError(f"Invalid type {type(vector)}")
        if isinstance(measure, str):
            measure = MeasureFactory.get_measure(measure)
        if not isinstance(measure, AbstractMeasure):
            raise TypeError(f"Invalid type {type(measure)}")
        return self._sim_search(vector, measure, k, **kwargs)

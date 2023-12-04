import uuid
import numpy as np
from typing import Optional, Union

from models.text import TextDoc, TextDocEmbedded
from models._types import ArrayLike


class Vector:

    def __init__(self,
                 embedding: Union[ArrayLike, TextDocEmbedded],
                 data: Optional[Union[str, TextDoc]] = None,
                 metadata: Optional[dict] = None):
        self.embedding = self.set_embedding(embedding)
        self.data = self.set_data(data)
        self.metadata = self.set_metadata(metadata)
        self.vec_id = uuid.uuid4()
        self.vector_dim = len(self.embedding)

    def set_embedding(self, value: Union[ArrayLike, TextDocEmbedded]) -> None:
        if isinstance(value, TextDocEmbedded):
            value = value.body
        if not isinstance(value, ArrayLike):
            raise TypeError(f"Invalid type {type(value)}. Must be ArrayLike.")
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return value

    def set_data(self, value: Optional[Union[str, TextDoc]]) -> None:
        if isinstance(value, TextDoc):
            value = value.body
        if not isinstance(value, (str, type(None), TextDoc)):
            raise TypeError(
                f"Invalid type {type(value)}. Must be type `str` or `TextDoc`.")
        return value

    def set_metadata(self, value: Optional[dict]) -> None:
        if not isinstance(value, (dict, type(None))):
            raise TypeError(
                f"Invalid type {type(value)}. Must be type `dict`.")
        return value

    def dot(self, other: 'Vector') -> float:
        return self.embedding.dot(other.embedding)

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(
            embedding=np.cross(self.embedding, other.embedding),
            data=self.data
        )

    def norm(self) -> float:
        return np.linalg.norm(self.embedding)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(
            embedding=self.embedding + other.embedding,
            data=self.data
        )

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(
            embedding=self.embedding - other.embedding,
            data=self.data
        )

    def __mul__(self, scalar: Union[float, int]) -> 'Vector':
        if not isinstance(scalar, (float, int)):
            raise TypeError((f"Invalid type {type(scalar)}. Scalar product expects a"
                             " scalar (float). Looking for dot() or cross()?"))
        return Vector(
            embedding=self.embedding * scalar,
            data=self.data
        )

    def __repr__(self):
        return f"<Vector {self.vec_id} ({self.vector_dim}){f' : {self.data}' if self.data else ''}>"

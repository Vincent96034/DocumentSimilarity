from typing import Union
import numpy as np

from models.text import TextDoc, TextDocEmbedded


"""Type for text-like objects. Used for input and output of
embedding models."""
TextLike = Union[str, TextDoc]


#! unused
"""Type for embedded text-like objects. Used for input and output of
embedding models."""
EmbeddedLike = Union[list, TextDocEmbedded, np.ndarray]


"""Type for array-like objects."""
ArrayLike = Union[list, np.ndarray]

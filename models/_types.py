from typing import Union
from models.text import TextDoc, TextDocEmbedded
import numpy as np


"""Type for text-like objects. Used for input and output of
embedding models."""
TextLike = Union[str, TextDoc]


#! unused
"""Type for embedded text-like objects. Used for input and output of
embedding models."""
EmbeddedLike = Union[list, TextDocEmbedded, np.ndarray]

from typing import List, Any
from abc import ABC, abstractmethod
import string

from models.text import TextDoc, TextDocEmbedded
from models._types import TextLike


class AbstractEmbedder(ABC):
    """Abstract base class for document embedders."""

    def fit(self, docs: List[TextLike]) -> None:
        """Fits the Embedder model to the given list of documents. Must be called
        before embedding any documents.

        Args:
            docs (List[TextLike]): A list of documents to fit the model on.
                Each document can be a string or an instance of the TextDoc class.
        """
        if not isinstance(docs, list):
            raise TypeError(
                '`docs` must be a list of `TextLike` documents.')

        def check_text_like(doc: Any) -> TextDoc:
            if isinstance(doc, list):
                raise TypeError(
                    'Nested list of documents is not supported.')
            if isinstance(doc, str):
                return TextDoc(body=doc)
            if not isinstance(doc, TextDoc):
                raise TypeError(
                    '`docs` must only include `TextLike` documents.')
            return doc

        docs = [check_text_like(doc) for doc in docs]
        return self._fit(docs)

    def embed(self, doc: TextLike) -> TextDocEmbedded:
        """Embeds a document into a vector representation.

        Args:
            doc (Union[TextDoc, str]): The document to be embedded.
                It can be either a `TextDoc` object or a string.

        Returns:
            TextDocEmbedded: The embedded document.
        """
        if isinstance(doc, list):
            # todo: support list of documents
            raise TypeError('List of documents is not supported.')
        if isinstance(doc, str):
            doc = TextDoc(body=doc)
        if not isinstance(doc, TextDoc):
            raise TypeError('Document must be of type `TextDoc` or `str`.')
        return self._embed(doc)

    def bulk_embed(self, docs: List[TextLike]) -> List[TextDocEmbedded]:
        """Embeds a list of documents into a vector representation.

        Args:
            docs (List[Union[TextDoc, str]]): A list of documents to be embedded.
                Each document can be either a `TextDoc` object or a string.

        Returns:
            List[TextDocEmbedded]: A list of embedded documents.
        """
        if not isinstance(docs, list):
            raise TypeError(
                '`docs` must be a list of `TextLike` documents.')
        for doc in docs:
            if isinstance(doc, list):
                raise TypeError(
                    'Nested list of documents is not supported.')
            if isinstance(doc, str):
                doc = TextDoc(body=doc)
            if not isinstance(doc, TextDoc):
                raise TypeError(
                    '`docs` must only include `TextLike` documents.')
        return [self.embed(doc) for doc in docs]

    def _clean_text(self, text: str) -> str:
        """Cleans the text by removing newline, tab, and carriage return characters,
        reducing multiple spaces to a single space, removing punctuation, and converting
        the text to lowercase.
        """
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        text = text.lower()
        return text

    @abstractmethod
    def _fit(self, docs: list) -> None:
        """Fits the embedder model to the given list of documents.

        Args:
            docs (list): A list of documents to fit the model on.
        """
        pass

    @abstractmethod
    def _embed(self, doc: TextDoc) -> TextDocEmbedded:
        """Embeds a text document into a vector representation.

        Args:
            doc (TextDoc): The text document to be embedded.

        Returns:
            TextDocEmbedded: The embedded document.
        """
        pass

    @abstractmethod
    def _preprocess(self, **kwargs):
        """Preprocesses the text document before embedding.

        Args:
            **kwargs: Additional keyword arguments for preprocessing.
        """
        pass

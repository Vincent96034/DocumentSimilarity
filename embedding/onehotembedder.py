from typing import List, Optional

from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

from embedding.embedder import AbstractEmbedder
from embedding.stopwords import stopwords_en
from models.text import TextDoc, TextDocEmbedded
from utils.logger_config import logger


class OneHotEmbedder(AbstractEmbedder):
    """Implements one-hot encoding for text embedding with optional dimensionality
    reduction using PCA.

    Attributes:
        corpus (dict): A dictionary to store the corpus of words and their corresponding
            indices.
        corpus_dim (int): The dimension of the corpus.
        vector_dim (Optional[int]): The dimension of the vector for PCA. If None, PCA is
            not applied.
        embedding_method (str): The embedding method to use. `one-hot` only indicates if a
            word is in the document or not. `additive` indicates the number of times a
            word appears in the document.
        _pca (Optional[PCA]): The PCA model for dimensionality reduction.
        _data (Optional[np.ndarray]): The document-term matrix used for PCA fitting.
    """

    def __init__(self, vector_dim: Optional[int] = None, embedding_method: str = "one-hot"):
        super().__init__()
        self.corpus = {}
        self.corpus_dim = 0
        self.vector_dim = vector_dim
        self.embedding_method = self._check_embedding_method(embedding_method)
        self._pca = None
        self._data = None

    def _fit(self, docs: List[TextDoc]) -> None:
        """Fits the embedder to the provided documents.
        This involves building the corpus and preparing the document-term matrix. If
        `vector_dim` is set, PCA is applied to reduce the dimensionality.

        Args:
            docs (List[TextDoc]): A list of `TextDoc` objects to fit the embedder.

        Returns:
            None
        """
        logger.info("Fitting embedder. Preprocessing documents...")
        docs = [self._preprocess(doc) for doc in tqdm(docs)]
        n_docs = len(docs)
        words = [word for doc in docs for word in doc]
        bow = list(set(words))
        for word in bow:
            if word not in self.corpus:
                self.corpus[word] = len(self.corpus)
        self.corpus_dim = len(self.corpus)
        logger.info(
            f"Corpus created with {self.corpus_dim} words.")
        # Creating document-term matrix for PCA
        if self.vector_dim:
            logger.info(
                f"Reducing vector dimensions: fitting PCA (n={self.vector_dim})")
            if self.vector_dim > n_docs:
                logger.warning(
                    (f'`vector_dim` ({self.vector_dim}) is greater than the number of '
                     f'documents ({n_docs}). Set `vector_dim` to {n_docs}.'))
                self.vector_dim = n_docs
            self._pca = PCA(n_components=self.vector_dim)
            doc_term_matrix = []
            for doc in docs:
                vector = self._create_vector(doc, method=self.embedding_method)
                doc_term_matrix.append(vector)
            doc_term_matrix = np.array(doc_term_matrix)
            self._pca.fit(doc_term_matrix)
            self._data = doc_term_matrix

    def _create_vector(self, doc: List[str], method="one-hot") -> List[int]:
        """Creates a vector representation of the given document.

        Args:
            doc (List[str]): A list of tokens representing the document.
            method (str): The embedding method to use. `one-hot` only indicates if a
                word is in the document or not. `additive` indicates the number of times a
                word appears in the document.

        Returns:
            List[int]: The vector representation of the document.
        """
        vector = [0] * self.corpus_dim
        for word in doc:
            if word in self.corpus:
                if method == "one-hot":
                    vector[self.corpus[word]] = 1
                elif method == "additive":
                    vector[self.corpus[word]] += 1
        return vector

    def _embed(self, doc: TextDoc) -> TextDocEmbedded:
        """Embeds a single document using one-hot encoding.

        Args:
            doc (TextDoc): The `TextDoc` object to embed.

        Returns:
            TextDocEmbedded: The embedded representation of the document.
        """
        doc = self._preprocess(doc)
        vector = self._create_vector(doc, method=self.embedding_method)
        return self._postprocess(vector)

    def _preprocess(self, doc: TextDoc) -> List[str]:
        """Preprocesses the document by cleaning, tokenizing, and removing stopwords.

        Args:
            doc (TextDoc): The `TextDoc` object to preprocess.

        Returns:
            List[str]: A list of tokens after preprocessing.
        """
        doc_clean = self._clean_text(doc.body)
        doc_clean = self._tokenize(doc_clean)
        doc_clean = [word for word in doc_clean if word not in stopwords_en]
        return doc_clean

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes the given text string.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        return text.split(" ")

    def _postprocess(self, vector: List[int]) -> TextDocEmbedded:
        """Postprocesses the vector, applying PCA if initialized.

        Args:
            vector (List[int]): The vector to postprocess.

        Returns:
            TextDocEmbedded: The postprocessed embedded representation.
        """
        if self._pca:
            vector = self._pca.transform([vector])[0]
        return TextDocEmbedded(body=vector)

    @staticmethod
    def _check_embedding_method(method: str) -> str:
        """Checks if the embedding method is valid.

        Args:
            method (str): The embedding method to check.

        Returns:
            str: The embedding method to use.
        """
        if method not in ["one-hot", "additive"]:
            logger.warning(
                f"Invalid embedding method `{method}`. Using `one-hot` instead.")
            return "one-hot"
        return method

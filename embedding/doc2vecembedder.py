from typing import List

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

from embedding.embedder import AbstractEmbedder
from embedding.stopwords import stopwords_en
from models.text import TextDoc, TextDocEmbedded
from utils.logger_config import logger


class Doc2VecEmbedder(AbstractEmbedder):
    """A class that represents a Doc2Vec Embedder.

    Attributes:
        vector_dim (int): The dimensionality of the embedding vectors.
        window (int): The maximum distance between the current and predicted word within a
            sentence.
        min_count (int): The minimum count of words to consider when training the model.
        workers (int): The number of worker threads to train the model.
        epochs (int): The number of iterations to train the model.

    Methods:
        _fit(docs: List[TextDoc]) -> None:
            Fits the embedder by preprocessing the documents and training the Doc2Vec model.

        save_model(path: str) -> None:
            Saves the model to the given path.

        _embed(doc: TextDoc) -> TextDocEmbedded:
            Embeds a document by inferring its vector representation using the trained model.

        _preprocess(doc: TextDoc) -> List[str]:
            Preprocesses the document by cleaning, tokenizing, and removing stopwords.

        _tokenize(text: str) -> List[str]:
            Tokenizes the given text string.

        _postprocess(vector: List[int]) -> TextDocEmbedded:
            Postprocesses the vector, applying PCA if initialized.
    """

    def __init__(self,
                 vector_dim: int = 150,
                 window: int = 5,
                 min_count: int = 1,
                 workers: int = 4,
                 epochs: int = 40):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def _fit(self, docs: List[TextDoc]) -> None:
        """Fits the embedder by preprocessing the documents and training the Doc2Vec
        model.

        Args:
            docs (List[TextDoc]): The list of `TextDoc` objects to fit the embedder.

        Returns:
            None
        """
        logger.info("Fitting embedder. Preprocessing documents...")
        docs = [self._preprocess(doc) for doc in tqdm(docs)]
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)])
                       for i, _d in enumerate(docs)]
        # Create and fit Doc2Vec model
        doc2vec_model = Doc2Vec(vector_size=self.vector_dim,
                                window=self.window,
                                min_count=self.min_count,
                                workers=self.workers,
                                epochs=self.epochs)
        doc2vec_model.build_vocab(tagged_data)
        logger.info("Fitting Doc2Vec model...")
        doc2vec_model.train(tagged_data,
                            total_examples=doc2vec_model.corpus_count,
                            epochs=doc2vec_model.epochs)
        self._model = doc2vec_model

    def save_model(self, path: str) -> None:
        """Saves the model to the given path.

        Args:
            path (str): The path to save the model.

        Returns:
            None
        """
        self._model.save(path + ".model")

    def _embed(self, doc: TextDoc) -> TextDocEmbedded:
        """Embeds a document by inferring its vector representation using the trained
        model.

        Args:
            doc (TextDoc): The `TextDoc` object to embed.

        Returns:
            TextDocEmbedded: The embedded representation of the document.
        """
        doc = self._preprocess(doc)
        vector = self._model.infer_vector(doc)
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
        """Postprocesses the vector. In this case, no postprocessing is done.

        Args:
            vector (List[int]): The vector to postprocess.

        Returns:
            TextDocEmbedded: The postprocessed embedded representation.
        """
        return TextDocEmbedded(body=vector)

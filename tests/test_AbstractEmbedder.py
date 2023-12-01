import pytest

from embedding.embedder import AbstractEmbedder
from models.text import TextDoc, TextDocEmbedded


class TestAbstractEmbedder(AbstractEmbedder):
    """Concrete subclass of AbstractEmbedder for testing purposes"""

    def _fit(self, docs):
        return docs

    def _embed(self, doc):
        return doc

    def _decode(self, doc):
        return doc

    def _preprocess(self, **kwargs):
        pass


@pytest.fixture
def test_embedder():
    return TestAbstractEmbedder()


@pytest.mark.parametrize("input_, e_", [
    ([TextDoc("ABC"), TextDoc("DEF")], None),
    ([TextDoc("ABC"), "DEF"], None),
    ([TextDoc("ABC"), 1], TypeError),
    ([TextDoc("ABC"), TextDoc("DEF")], None),
    ([TextDoc("ABC"), "DEF"], None),
    (["ABC", 1.0], TypeError),
    ("ABC", TypeError),
    (TextDoc("ABC"), TypeError),
    (1.0, TypeError),
])
def test_fit_IO(test_embedder, input_, e_):
    if e_ == TypeError:
        with pytest.raises(e_):
            test_embedder.fit(input_)
    else:
        output_ = test_embedder.fit(input_)
        assert isinstance(output_, list)
        for doc in output_:
            assert isinstance(doc, TextDoc)


@pytest.mark.parametrize("input_, e_", [
    (TextDoc("ABC"), None),
    ("ABC", None),
    (1.0, TypeError),
    ([TextDoc("ABC"), TextDoc("DEF")], TypeError),
    ([1, "ABC"], TypeError)
])
def test_embed_IO(test_embedder, input_, e_):

    if e_ == TypeError:
        with pytest.raises(e_):
            test_embedder.embed(input_)
    else:
        # Note: In the actual implementation, the output is a TextDocEmbedded
        # object, but for testing purposes of the AbstractClass, we will just
        # return the input.
        output_ = test_embedder.embed(input_)
        assert isinstance(output_, TextDoc)


@pytest.mark.parametrize("input_, e_", [
    (TextDocEmbedded([1, 1, 1]), None),
    (TextDoc("ABC"), TypeError),
    ([1, 1, 1, 1], TypeError),
    ("sdfs", TypeError),
    (1.0, TypeError),
    ([TextDoc("ABC"), TextDoc("DEF")], TypeError),
    ([1, "ABC"], TypeError)
])
def test_decode_IO(test_embedder, input_, e_):

    if e_ == TypeError:
        with pytest.raises(e_):
            test_embedder.decode(input_)
    else:
        # Note: In the actual implementation, the output is a TextDoc object,
        # but for testing purposes of the AbstractClass, we will just return
        # the input.
        output_ = test_embedder.decode(input_)
        assert isinstance(output_, TextDocEmbedded)

from collections.abc import Iterable, Iterator
from types import TracebackType

from nltk import CoreNLPParser as NLTKParser

from architxt.tree import Tree

from . import Parser

__all__ = ['CoreNLPParser']


class CoreNLPParser(Parser):
    def __init__(
        self,
        *,
        corenlp_url: str,
    ) -> None:
        """
        Create a CoreNLP parser.

        :param corenlp_url: The URL of the CoreNLP server.
        """
        self.corenlp = NLTKParser(url=corenlp_url)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.corenlp.session.close()

    def raw_parse(self, sentences: Iterable[str], *, language: str) -> Iterator[Tree]:
        for tree in self.corenlp.raw_parse_sents(sentences, properties={'tokenize.language': language}):
            # CoreNLP return a list of candidates tree, we only select the first one.
            # A parse tree may contain multiple sentence subtrees we select only one and convert it into a tree.
            yield Tree.convert(next(tree)[0])

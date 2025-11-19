from __future__ import annotations

from typing import TYPE_CHECKING

import more_itertools
from nltk import CoreNLPParser as NLTKParser

from architxt.tree import Tree

from . import Parser

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import TracebackType

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

        # patch CoreNLP client to use our Tree class instead of the NLTK one
        self.corenlp.make_tree = lambda result: Tree.fromstring(result["parse"])

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.corenlp.session.close()

    def raw_parse(self, sentences: Iterable[str], *, language: str, batch_size: int = 128) -> Iterator[Tree]:
        for batch in more_itertools.chunked(sentences, batch_size):
            for tree in self.corenlp.raw_parse_sents(batch, properties={'tokenize.language': language}):
                # CoreNLP return a list of candidates tree, we only select the first one.
                # A parse tree may contain multiple sentence subtrees we select only one and convert it into a tree.
                yield next(tree)[0]

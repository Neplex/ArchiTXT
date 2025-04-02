from collections.abc import Iterable, Iterator
from types import TracebackType

import benepar  # noqa: F401
import spacy
from spacy import Language

from architxt.tree import Tree

from . import Parser

__all__ = ['BeneparParser']

DEFAULT_BENEPAR_MODELS = {
    'English': 'benepar_en3',
    'Chinese': 'benepar_zh2',
    'Arabic': 'benepar_ar2',
    'German': 'benepar_de2',
    'Basque': 'benepar_eu2',
    'French': 'benepar_fr2',
    'Hebrew': 'benepar_he2',
    'Hungarian': 'benepar_hu2',
    'Korean': 'benepar_ko2',
    'Polish': 'benepar_pl2',
    'Swedish': 'benepar_sv2',
}


class BeneparParser(Parser):
    def __init__(
        self,
        *,
        spacy_models: dict[str, str],
        benepar_models: dict[str, str] | None = None,
    ) -> None:
        """
        Create a benepar parser.

        :param spacy_models: The name of the SpaCy models for each language.
        :param benepar_models: The name of the SpaCy model to use.
        """
        self.spacy_models = spacy_models
        self.benepar_models = benepar_models or DEFAULT_BENEPAR_MODELS
        self.__models: dict[str, Language] = {}

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.__models.clear()

    def _get_model(self, language: str) -> Language:
        if language not in self.__models:
            nlp = spacy.load(self.spacy_models[language], disable={'ner', 'textcat', 'lemmatizer', 'tagger'})
            nlp.add_pipe('benepar', config={'model': self.benepar_models[language]})
            self.__models[language] = nlp

        return self.__models[language]

    def raw_parse(self, sentences: Iterable[str], *, language: str, batch_size: int = 128) -> Iterator[Tree]:
        nlp = self._get_model(language)

        for doc in nlp.pipe(sentences, batch_size=batch_size):
            sent = next(doc.sents)
            tree = Tree.fromstring(sent._.parse_string)
            tree.set_label('SENT')
            yield tree

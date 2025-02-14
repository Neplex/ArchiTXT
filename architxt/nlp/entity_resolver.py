import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import TracebackType
from typing import cast

import spacy
from googletrans import Translator
from scispacy.linking import EntityLinker
from spacy.language import Doc, Language
from unidecode import unidecode


class EntityResolver(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def __call__(self, texts: Iterable[str]) -> Iterable[str]: ...


class ScispacyResolver(EntityResolver):
    def __init__(
        self,
        model: Language | str = 'en_core_sci_sm',
        *,
        kb_name: str = 'umls',
        cleanup: bool = False,
        translate: bool = False,
        batch_size: int = 8,
    ) -> None:
        """
        An entity resolver based on SciSpaCy entity linker.

        :param model: The SpaCy model to use.
        :param kb_name: The name of the knowledge base to use: `umls`, `mesh`, `rxnorm`, `go`, or `hpo`.
        :param cleanup: True if the resolved text should be uniformized.
        :param translate: True if the text should be translated if it does not correspond to the model language.
        :param batch_size: Number of texts to process in parallel (useful for large corpora).
        """
        self.nlp = spacy.load(model) if isinstance(model, str) else model
        self.translate = translate
        self.cleanup = cleanup
        self.batch_size = batch_size
        self.exit_stack = contextlib.AsyncExitStack()
        self.kb_name = kb_name

        linker_config = {"resolve_abbreviations": True, "linker_name": self.kb_name}
        linker = self.nlp.add_pipe("scispacy_linker", config=linker_config)
        self.linker = cast(EntityLinker, linker)

    async def __aenter__(self) -> 'ScispacyResolver':
        if self.translate:
            translator = Translator(list_operation_max_concurrency=self.batch_size)
            self.translator = await self.exit_stack.enter_async_context(translator)

        return self

    async def __aexit__(
        self, exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType
    ) -> None:
        await self.exit_stack.aclose()

    @property
    def language(self) -> str:
        return self.nlp.lang

    @property
    def name(self) -> str:
        return self.kb_name

    async def _translate(self, texts: list[str]) -> list[str]:
        """
        Translate texts in batch asynchronously.
        Uses an existing translator if available, otherwise creates a temporary one.
        """
        if not self.translator:
            async with Translator(list_operation_max_concurrency=self.batch_size) as temp_translator:
                translations = await temp_translator.translate(texts, dest=self.language)
        else:
            translations = await self.translator.translate(texts, dest=self.language)

        return [t.text for t in translations]

    def _resolve(self, document: Doc) -> Doc:
        """Resolve entity names using SciSpaCy entity linker."""
        for entity in document.ents:
            if entity._.kb_ents:
                cui = entity._.kb_ents[0][0]
                resolved_text = self.linker.kb.cui_to_entity[cui].canonical_name
                return self.nlp(resolved_text, disable=['ner'])

        return document

    def _cleanup_string(self, document: Doc) -> str:
        """
        Cleanup text to uniformize it.
        :param document: The text document to clean up.
        :return: The uniformized text.
        """
        if not self.cleanup:
            return document.text

        text = ' '.join(
            token.lemma_.lower() for token in document if (token.is_alpha or token.is_digit) and not token.is_stop
        )

        return unidecode(text) if text else ""

    async def __call__(self, texts: Iterable[str]) -> Iterable[str]:
        if self.translate:
            texts = await self._translate(list(texts))

        return (self._cleanup_string(self._resolve(doc)) for doc in self.nlp.pipe(texts, batch_size=self.batch_size))

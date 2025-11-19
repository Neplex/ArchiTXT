from __future__ import annotations

import abc
import contextlib
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from aiostream import pipe, stream
from googletrans import Translator
from scispacy.candidate_generation import CandidateGenerator
from typing_extensions import Self
from unidecode import unidecode

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable
    from types import TracebackType

    from architxt.nlp.model import AnnotatedSentence, Entity


class EntityResolver(AbstractAsyncContextManager):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    async def __call__(self, entity: Entity) -> Entity: ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        pass

    async def batch(
        self,
        entities: Iterable[Entity] | AsyncIterable[Entity],
        *,
        batch_size: int = 16,
    ) -> AsyncIterator[Entity]:
        entity_stream = stream.iterate(entities) | pipe.amap(self.__call__, task_limit=batch_size)

        async with entity_stream.stream() as streamer:
            async for entity in streamer:
                yield entity

    async def batch_sentences(
        self,
        sentences: Iterable[AnnotatedSentence] | AsyncIterable[AnnotatedSentence],
        *,
        batch_size: int = 16,
    ) -> AsyncIterator[AnnotatedSentence]:
        async def _resolve(sentence: AnnotatedSentence) -> AnnotatedSentence:
            sentence.entities = [entity async for entity in self.batch(sentence.entities, batch_size=batch_size)]
            return sentence

        sentence_stream = stream.iterate(sentences) | pipe.amap(_resolve, task_limit=1)
        async with sentence_stream.stream() as streamer:
            async for sent in streamer:
                yield sent


class ScispacyResolver(EntityResolver):
    def __init__(
        self,
        *,
        kb_name: str = 'umls',
        cleanup: bool = False,
        translate: bool = False,
        threshold: float = 0.7,
        resolve_text: bool = True,
    ) -> None:
        """
        Resolve entities using the SciSpaCy entity linker.

        :param kb_name: The name of the knowledge base to use: `umls`, `mesh`, `rxnorm`, `go`, or `hpo`.
        :param cleanup: True if the resolved text should be uniformized.
        :param translate: True if the text should be translated if it does not correspond to the model language.
        :param threshold : The threshold that an entity candidate must reach to be considered.
        :param resolve_text: True if the resolver should return the canonical name instead of the identifier
        """
        self.translate = translate
        self.cleanup = cleanup
        self.threshold = threshold
        self.kb_name = kb_name
        self.resolve_text = resolve_text
        self.translator: Translator | None = None

        self.exit_stack = contextlib.AsyncExitStack()
        self.candidate_generator = CandidateGenerator(name=self.kb_name)

    async def __aenter__(self) -> Self:
        if self.translate:
            translator = Translator()
            self.translator = await self.exit_stack.enter_async_context(translator)

        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.exit_stack.aclose()

    @property
    def name(self) -> str:
        return self.kb_name

    async def _translate(self, text: str) -> str:
        """
        Translate text asynchronously.

        Use an existing translator if available, otherwise creates a temporary one.
        """
        if not self.translator:
            async with Translator() as temp_translator:
                translation = await temp_translator.translate(text, dest="en")
        else:
            translation = await self.translator.translate(text, dest="en")

        return translation.text

    def _cleanup_string(self, text: str) -> str:
        """
        Cleanup text to uniformize it.

        :param text: The text document to clean up.
        :return: The uniformized text.
        """
        if text and self.cleanup:
            text = unidecode(text.lower())

        return text

    def _resolve(self, text: str) -> str:
        """Resolve entity names using SciSpaCy entity linker."""
        candidates = self.candidate_generator([text], 10)[0]
        best_candidate = None
        best_candidate_score = 0

        for candidate in candidates:
            if (score := max(candidate.similarities, default=0)) > self.threshold and score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = score

        if not best_candidate:
            return text

        if self.resolve_text:
            return self.candidate_generator.kb.cui_to_entity[best_candidate.concept_id].canonical_name

        return best_candidate.concept_id

    async def __call__(self, entity: Entity) -> Entity:
        if self.translate:
            value = await self._translate(entity.value)
        else:
            value = entity.value

        entity.value = self._cleanup_string(self._resolve(value))

        return entity

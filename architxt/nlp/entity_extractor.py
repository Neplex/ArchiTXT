from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiostream import pipe, stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable

    from architxt.nlp.model import AnnotatedSentence

__all__ = ['EntityExtractor']


class EntityExtractor(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, sentence: str) -> AnnotatedSentence: ...

    async def batch(
        self,
        sentences: Iterable[str] | AsyncIterable[str],
    ) -> AsyncIterator[AnnotatedSentence]:
        sentence_stream = stream.iterate(sentences) | pipe.map(self.__call__)

        async with sentence_stream.stream() as streamer:
            async for sentence in streamer:
                yield sentence

    async def enrich(
        self,
        sentences: Iterable[AnnotatedSentence] | AsyncIterable[AnnotatedSentence],
    ) -> AsyncIterator[AnnotatedSentence]:
        def _enrich_sentence(annotated: AnnotatedSentence) -> AnnotatedSentence:
            new_entities = self(annotated.txt).entities
            annotated.entities.extend(new_entities)
            return annotated

        sentence_stream = stream.iterate(sentences) | pipe.map(_enrich_sentence)

        async with sentence_stream.stream() as streamer:
            async for sentence in streamer:
                yield sentence

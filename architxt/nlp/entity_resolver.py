from __future__ import annotations

import abc
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from aiostream import pipe, stream
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable
    from types import TracebackType

    from architxt.nlp.model import AnnotatedSentence, Entity

__all__ = ['EntityResolver']


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

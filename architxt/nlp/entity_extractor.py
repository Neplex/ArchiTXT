from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiostream import pipe, stream

from architxt.nlp.model import AnnotatedSentence, Entity

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable

    from flair.data import Sentence
    from spacy.tokens import Doc

SPACY_DISABLED_PIPELINES = {'parser', 'senter', 'sentencizer', 'textcat', 'lemmatizer', 'tagger'}


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


class SpacyEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        import spacy

        self.nlp = spacy.load(model_name, disable=SPACY_DISABLED_PIPELINES)

    @staticmethod
    def _doc_to_annotated(doc: Doc) -> AnnotatedSentence:
        entities = [
            Entity(
                name=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                id=f"{ent.label_}_{ent.start_char}_{ent.end_char}",
                value=ent.text,
            )
            for ent in doc.ents
        ]
        return AnnotatedSentence(txt=doc.text, entities=entities, rels=[])

    def __call__(self, sentence: str) -> AnnotatedSentence:
        doc = self.nlp(sentence)
        return self._doc_to_annotated(doc)

    async def batch(
        self,
        sentences: Iterable[str] | AsyncIterable[str],
        *,
        batch_size: int = 128,
    ) -> AsyncIterator[AnnotatedSentence]:
        sentence_stream = (
            stream.iterate(sentences)
            | pipe.chunks(batch_size)
            | pipe.flatmap(self.nlp.pipe)
            | pipe.map(self._doc_to_annotated)
        )

        async with sentence_stream.stream() as streamer:
            async for sentence in streamer:
                yield sentence


class FlairEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str = "ner") -> None:
        from flair.models import SequenceTagger

        self.tagger = SequenceTagger.load(model_name)

    @staticmethod
    def _sentence_to_annotated(sentence: Sentence) -> AnnotatedSentence:
        entities = [
            Entity(
                name=span.tag,
                start=span.start_position,
                end=span.end_position,
                id=f"{span.tag}_{span.start_position}_{span.end_position}",
                value=span.text,
            )
            for span in sentence.get_spans('ner')
        ]
        return AnnotatedSentence(txt=sentence.to_plain_string(), entities=entities, rels=[])

    def __call__(self, sentence: str) -> AnnotatedSentence:
        from flair.data import Sentence

        flair_sentence = Sentence(sentence)
        self.tagger.predict(flair_sentence)

        return self._sentence_to_annotated(flair_sentence)

    async def batch(
        self,
        sentences: Iterable[str] | AsyncIterable[str],
        *,
        batch_size: int = 128,
    ) -> AsyncIterator[AnnotatedSentence]:
        from flair.data import Sentence

        entity_stream = (
            stream.iterate(sentences)
            | pipe.map(Sentence)
            | pipe.chunks(batch_size)
            | pipe.flatmap(self.tagger.predict)
            | pipe.map(self._sentence_to_annotated)
        )

        async with entity_stream.stream() as streamer:
            async for doc in streamer:
                yield doc

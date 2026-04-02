from __future__ import annotations

from typing import TYPE_CHECKING

import spacy
from aiostream import pipe, stream

from architxt.nlp.entity_extractor import EntityExtractor
from architxt.nlp.model import AnnotatedSentence, Entity

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable

    from spacy.tokens import Doc

__all__ = ['SpacyEntityExtractor']

SPACY_DISABLED_PIPELINES = {'parser', 'senter', 'sentencizer', 'textcat', 'lemmatizer', 'tagger'}


class SpacyEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
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

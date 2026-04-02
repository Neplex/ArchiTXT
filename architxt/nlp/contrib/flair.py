from __future__ import annotations

from typing import TYPE_CHECKING

from aiostream import pipe, stream

from architxt.nlp.entity_extractor import EntityExtractor
from architxt.nlp.model import AnnotatedSentence, Entity

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable


try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError as error:
    msg = f"The '{__name__}' contrib module requires Flair. Install it with: pip install architxt[flair]"
    raise ImportError(msg) from error

__all__ = ['FlairEntityExtractor']


class FlairEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str = "ner") -> None:
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
        flair_sentence = Sentence(sentence)
        self.tagger.predict(flair_sentence)

        return self._sentence_to_annotated(flair_sentence)

    async def batch(
        self,
        sentences: Iterable[str] | AsyncIterable[str],
        *,
        batch_size: int = 128,
    ) -> AsyncIterator[AnnotatedSentence]:
        entity_stream = (
            stream.iterate(sentences)
            | pipe.map(Sentence)
            | pipe.chunks(batch_size)
            | pipe.action(self.tagger.predict)
            | pipe.flatten()
            | pipe.map(self._sentence_to_annotated)
        )

        async with entity_stream.stream() as streamer:
            async for doc in streamer:
                yield doc

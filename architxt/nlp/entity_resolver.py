import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import TracebackType

from googletrans import Translator
from scispacy.candidate_generation import CandidateGenerator
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
        *,
        kb_name: str = 'umls',
        cleanup: bool = False,
        translate: bool = False,
        batch_size: int = 8,
        threshold: float = 0.7,
        resolve_text: bool = True,
    ) -> None:
        """
        An entity resolver based on SciSpaCy entity linker.

        :param kb_name: The name of the knowledge base to use: `umls`, `mesh`, `rxnorm`, `go`, or `hpo`.
        :param cleanup: True if the resolved text should be uniformized.
        :param translate: True if the text should be translated if it does not correspond to the model language.
        :param batch_size: Number of texts to process in parallel (useful for large corpora).
        :param threshold : The threshold that an entity candidate must reach to be considered.
        :param resolve_text: True if the resolver should return the canonical name instead of the identifier
        """
        self.translate = translate
        self.cleanup = cleanup
        self.batch_size = batch_size
        self.threshold = threshold
        self.kb_name = kb_name
        self.resolve_text = resolve_text

        self.exit_stack = contextlib.AsyncExitStack()
        self.candidate_generator = CandidateGenerator(name=self.kb_name)

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
    def name(self) -> str:
        return self.kb_name

    async def _translate(self, texts: list[str]) -> list[str]:
        """
        Translate texts in batch asynchronously.
        Uses an existing translator if available, otherwise creates a temporary one.
        """
        if not self.translator:
            async with Translator(list_operation_max_concurrency=self.batch_size) as temp_translator:
                translations = await temp_translator.translate(texts, dest="en")
        else:
            translations = await self.translator.translate(texts, dest="en")

        return [t.text for t in translations]

    def _cleanup_string(self, text: str) -> str:
        """
        Cleanup text to uniformize it.
        :param text: The text document to clean up.
        :return: The uniformized text.
        """
        if text and self.cleanup:
            text = unidecode(text.lower())

        return text

    def _resolve(self, mention_texts: list[str]) -> Iterable[str]:
        """Resolve entity names using SciSpaCy entity linker."""
        for mention, candidates in zip(mention_texts, self.candidate_generator(mention_texts, 10)):
            best_candidate = None
            best_candidate_score = 0

            for candidate in candidates:
                if (score := max(candidate.similarities, default=0)) > self.threshold and score > best_candidate_score:
                    best_candidate = candidate
                    best_candidate_score = score

            if not best_candidate:
                yield mention

            elif self.resolve_text:
                yield self.candidate_generator.kb.cui_to_entity[best_candidate.concept_id].canonical_name

            else:
                yield best_candidate.concept_id

    async def __call__(self, texts: Iterable[str]) -> Iterable[str]:
        texts = list(texts)

        if self.translate:
            texts = await self._translate(texts)

        return map(self._cleanup_string, self._resolve(texts))

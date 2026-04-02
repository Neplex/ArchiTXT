from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from googletrans import Translator
from unidecode import unidecode

from architxt.nlp.entity_resolver import EntityResolver

if TYPE_CHECKING:
    from types import TracebackType

    from architxt.nlp.model import Entity


try:
    from scispacy.candidate_generation import CandidateGenerator
except ImportError as error:
    msg = f"The '{__name__}' contrib module requires SciSpaCy. Install it with: pip install architxt[scispacy]"
    raise ImportError(msg) from error

__all__ = ['ScispacyResolver']


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

    async def __aenter__(self) -> ScispacyResolver:  # noqa: PYI034
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

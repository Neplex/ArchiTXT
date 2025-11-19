from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from unidecode import unidecode

from architxt.nlp.model import Entity, Relation

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

__all__ = ['split_entities', 'split_relations', 'split_sentences']


def split_sentences(text: str) -> list[str]:
    r"""
    Remove Unicode and split the input text into sentences based on the line breaks.

    It is common for brat annotation formats to have one sentence per line.

    :param text: The input text to be split into sentences.
    :return: A list of sentences split by line breaks with Unicode removed.

    >>> split_sentences("This is à test\nAnothér-test here")
    ['This is a test', 'Another-test here']

    """
    return unidecode(text).split('\n')


def split_entities(entities: Iterable[Entity], sentences: Sequence[str]) -> Generator[list[Entity], None, None]:
    """
    Split a list of `Entity` objects based on their occurrence in different sentences.

    Entities are assigned to sentences based on their start and end positions. The function
    returns a generator of lists, where each list contains the entities corresponding to a
    specific sentence, with the entity positions adjusted to be relative to the sentence.

    :param entities: An iterable of `Entity` objects, each representing a named entity with
                     start and end positions relative to the entire text.
    :param sentences: A sequence of sentences corresponding to the text from which the entities are extracted.

    :yield: A list of `Entity` objects for each sentence, with entity positions relative to
            that sentence.

    >>> e1 = Entity(name="Entity1", start=0, end=5, id="E1", value="x")
    >>> e2 = Entity(name="Entity2", start=6, end=15, id="E2", value="y")
    >>> e3 = Entity(name="Entity3", start=21, end=25, id="E3", value="z")
    >>> result = list(split_entities([e1, e2, e3], ["Hello world.", "This is a test."]))
    >>> len(result)
    2
    >>> len(result[0])
    1
    >>> len(result[1])
    2
    >>> result[0][0].name == "Entity1"
    True
    >>> result[1][0].name == "Entity2"
    True
    >>> result[1][1].name == "Entity3"
    True

    """
    # Sort entities by their start position
    entities = sorted(entities, key=lambda ent: (ent.start, ent.end))

    ent_i = 0  # Index to track the current entity
    sent_i = 0  # Index to track the current sentence
    start = 0  # Cumulative start index of the current sentence within the whole text

    # Iterate through each sentence
    while sent_i < len(sentences):
        sent_entities = []
        end = start + len(sentences[sent_i])  # The end index of the current sentence

        # Gather entities that belong to the current sentence
        while ent_i < len(entities) and entities[ent_i].end <= end:
            entity = entities[ent_i]

            # Calculate entity start and end positions relative to the current sentence
            ent_start = max(entity.start - start, 0)
            ent_end = min(entity.end - start, len(sentences[sent_i]))
            ent_i += 1

            # Add the entity to the list of entities for this sentence
            try:
                sent_entities.append(
                    Entity(start=ent_start, end=ent_end, name=entity.name, id=entity.id, value=entity.value)
                )
            except ValueError as error:
                warnings.warn(str(error))

        # Update the start position for the next sentence
        start += len(sentences[sent_i]) + 1  # +1 accounts for the space or punctuation between sentences
        sent_i += 1

        # Yield the entities corresponding to the current sentence
        yield sent_entities


def split_relations(relations: Iterable[Relation], entities: Sequence[Sequence[Entity]]) -> list[list[Relation]]:
    """
    Split relations into sentence-specific relationships.

    It maps the entity IDs to their indices within the corresponding sentence's entities.

    :param relations: An iterable of `Relation`.
    :param entities: A sequence of sequences, where each inner sequence contains `Entity` objects
                     corresponding to entities in a sentence.

    :return: A list of lists. Each inner list corresponds to a sentence and contains `Relation` objects
             for that sentence.

    >>> e1 = Entity(name="Entity1", start=0, end=1, id="E1", value="1")
    >>> e2 = Entity(name="Entity2", start=2, end=3, id="E2", value="2")
    >>> e3 = Entity(name="Entity3", start=4, end=5, id="E3", value="3")
    >>> e4 = Entity(name="Entity4", start=6, end=7, id="E4", value="4")
    >>> r1 = Relation(src="E1", dst="E2", name="relates_to")
    >>> r2 = Relation(src="E3", dst="E4", name="belongs_to")
    >>> result = split_relations([r1, r2], [[e1, e2], [e3, e4]])
    >>> len(result)
    2
    >>> result[0][0] == r1
    True
    >>> result[1][0] == r2
    True

    """
    # Initialize an empty list of relationships for each sentence
    relationship: list[list[Relation]] = [[] for _ in range(len(entities))]

    # Create a dictionary of entity indices for each sentence for faster lookups
    entity_index_map = [{entity.id: entity for entity in sentence_entities} for sentence_entities in entities]

    # Iterate through each relation and map it to the corresponding sentence and entity indices
    for rel in relations:
        # Find the sentence that contains both the source and destination entities
        sent_i: int | None = None

        for i, entity_map in enumerate(entity_index_map):
            if rel.src in entity_map and rel.dst in entity_map:
                sent_i = i
                break

        # If the relation belongs to a valid sentence, append it to the relationships
        if sent_i is not None:
            relationship[sent_i].append(rel)

    return relationship

"""
Dataset loader for BRAT (BRAT Rapid Annotation Tool) format
"""

from collections.abc import Generator, Iterable
from pathlib import Path

from bratlib.data import BratDataset, BratFile
from bratlib.data.annotation_types import Entity as BratEntity
from bratlib.data.annotation_types import Relation as BratRelation
from tqdm import tqdm

from architxt.nlp.model import AnnotatedSentence, Entity, Relation
from architxt.nlp.utils import split_entities, split_relations, split_sentences

__all__ = ['load_brat_dataset']


def convert_brat_entities(
    entities: Iterable[BratEntity],
    *,
    allow_list: set[str] | None = None,
    mapping: dict[str, str] | None = None,
) -> Generator[Entity, None, None]:
    """
    Converts a list of `BratEntity` objects into `Entity` objects, while filtering out certain types of tags.

    :param entities: An iterable of `BratEntity` objects to convert.
    :param allow_list: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param mapping: A dictionary mapping entity names to new values. If None, no mapping is applied.
    :return: A generator yielding `Entity` objects.

    Example:
    >>> ents = [
    ...     BratEntity(spans=[(0, 5)], tag="person", mention="E1"),
    ...     BratEntity(spans=[(10, 15)], tag="FREQ", mention="E2"),
    ...     BratEntity(spans=[(20, 25)], tag="MOMENT", mention="E3")
    ... ]
    >>> ents = list(convert_brat_entities(ents, allow_list={"MOMENT"}, mapping={"FREQ": "FREQUENCE"}))
    >>> len(ents)
    2
    >>> print(ents[0].name)
    PERSON
    >>> print(ents[1].name)
    FREQUENCE
    """
    allow_list = allow_list or set()
    mapping = mapping or {}

    for brat_entity in entities:
        # Start and end positions based on the spans of the entity
        start = brat_entity.spans[0][0]
        end = brat_entity.spans[-1][-1]

        # Rename tag if needed
        tag = brat_entity.tag.upper()
        tag = mapping.get(tag, tag)

        # Generate the identity of the entity based on its spans
        identity = tuple(brat_entity.spans)

        # Filter out entities with specific tags
        if tag not in allow_list:
            yield Entity(name=tag, start=start, end=end, id=str(identity))


def convert_brat_relations(
    relations: Iterable[BratRelation],
    *,
    allow_list: set[str] | None = None,
    mapping: dict[str, str] | None = None,
) -> Generator[Relation, None, None]:
    """
    Converts a list of `BratRelation` objects into `Relation` objects while filtering out certain types of relations.

    :param relations: An iterable of `BratRelation` objects to convert.
    :param allow_list: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :return: A generator yielding `Relation` objects.

    Example:
    >>> rels = [
    ...     BratRelation(arg1=BratEntity(spans=[(0, 5)], tag='X', mention='E1'), arg2=BratEntity(spans=[(10, 15)], tag='Y', mention='E2'), relation="part-of"),
    ...     BratRelation(arg1=BratEntity(spans=[(20, 25)], tag='X', mention='E3'), arg2=BratEntity(spans=[(30, 35)], tag='Z', mention='E3'), relation="TEMPORALITY")
    ... ]
    >>> rels = list(convert_brat_relations(rels, allow_list={"TEMPORALITY"}))
    >>> len(rels)
    1
    >>> print(rels[0].name)
    PART-OF
    """
    allow_list = allow_list or set()
    mapping = mapping or {}

    for brat_relation in relations:
        src = str(tuple(brat_relation.arg1.spans))
        dst = str(tuple(brat_relation.arg2.spans))

        # Rename relation if needed
        relation = brat_relation.relation.upper()
        relation = mapping.get(relation, relation)

        # Filter out specific relation types
        if relation not in allow_list and 'INCERTAIN' not in relation:
            yield Relation(src=src, dst=dst, name=relation)


def convert_brat_file(
    brat_file: BratFile,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
) -> Generator[AnnotatedSentence, None, None]:
    """
    Converts a BratFile object into annotated sentences, filtering and mapping entities and relations as specified.

    :param brat_file: A `BratFile` object containing the .txt and .ann file data.
    :param entities_filter: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param entities_mapping: A dictionary mapping entity names to new values. If None, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :return: A generator yielding `AnnotatedSentence` objects for each sentence in the text.
    """
    file_path = Path(brat_file.txt_path)
    text = file_path.read_text(encoding='utf-8')

    # Split the text into sentences
    sentences = list(split_sentences(text))

    # Convert and filter entities, split by sentences
    entities = list(
        split_entities(
            convert_brat_entities(brat_file.entities, allow_list=entities_filter, mapping=entities_mapping), sentences
        )
    )

    # Convert and filter relations, split by entities
    relationships = split_relations(
        convert_brat_relations(brat_file.relations, allow_list=relations_filter, mapping=relations_mapping), entities
    )

    # Yield AnnotatedSentence objects for each sentence with its corresponding entities and relations
    for sentence, sentence_entities, sentence_relations in zip(sentences, entities, relationships, strict=False):
        if sentence and sentence_entities:  # Yield only non-empty sentences
            yield AnnotatedSentence(sentence, sentence_entities, sentence_relations)


def load_brat_dataset(
    path: Path,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
) -> Generator[AnnotatedSentence, None, None]:
    dataset: BratDataset = BratDataset.from_directory(path.absolute())
    brat_file: BratFile

    for brat_file in (pbar := tqdm(dataset, total=len(dataset.brat_files))):
        file_path = Path(brat_file.txt_path)
        pbar.set_description(f'Load {file_path.name}')

        yield from convert_brat_file(
            brat_file,
            entities_filter=entities_filter,
            relations_filter=relations_filter,
            entities_mapping=entities_mapping,
            relations_mapping=relations_mapping,
        )

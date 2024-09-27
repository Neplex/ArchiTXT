from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

import requests.exceptions
from bratlib.data import BratDataset, BratFile
from bratlib.data.annotation_types import Entity as BratEntity
from bratlib.data.annotation_types import Relation as BratRelation
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm
from unidecode import unidecode

from architxt.model import AnnotatedSentence, Entity, NodeType, Relation
from architxt.tree import ParentedTree, enrich_tree, fix_all_coord, reduce_all


def split_sentences(text):
    text = unidecode(text)
    return text.split('\n')


def convert_brat_file(brat_file: BratFile) -> Generator[AnnotatedSentence]:
    file_path = Path(brat_file.txt_path)

    text = file_path.read_text(encoding='utf-8')

    sentences = list(split_sentences(text))
    entities = list(split_entities(convert_brat_entities(brat_file.entities), sentences))
    relationships = split_relations(convert_brat_relations(brat_file.relations), entities)

    for sentence, entities, rels in zip(sentences, entities, relationships):
        if sentence:
            yield AnnotatedSentence(sentence, entities, rels)


def convert_brat_entities(entities: Iterable[BratEntity]) -> Generator[Entity]:
    for brat_entity in entities:
        start = brat_entity.spans[0][0]
        end = brat_entity.spans[-1][-1]
        tag = brat_entity.tag.upper()
        identity = tuple(brat_entity.spans)

        if tag == 'FREQ':
            tag = 'FREQUENCE'

        if tag not in ['MOMENT', 'DUREE', 'DATE']:
            yield Entity(name=tag, start=start, end=end, id=str(identity))


def split_entities(entities: Iterable[Entity], sentences: Sequence[str]) -> Generator[list[Entity], None, None]:
    """
    Splits a list of `Entity` objects based on their occurrence in different sentences.

    Entities are assigned to sentences based on their start and end positions. The function
    returns a generator of lists, where each list contains the entities corresponding to a
    specific sentence, with the entity positions adjusted to be relative to the sentence.

    :param entities: An iterable of `Entity` objects, each representing a named entity with
                     start and end positions relative to the entire text.
    :param sentences: A sequence of sentences corresponding to the text from which the entities are extracted.

    :yield: A list of `Entity` objects for each sentence, with entity positions relative to
            that sentence.

    Example:
    >>> e1 = Entity(name="Entity1", start=0, end=5, id="E1")
    >>> e2 = Entity(name="Entity2", start=6, end=15, id="E2")
    >>> e3 = Entity(name="Entity3", start=21, end=25, id="E3")
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
    entities = sorted(entities, key=lambda ent: ent.start)

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

            # Add the entity to the list of entities for this sentence
            sent_entities.append(Entity(start=ent_start, end=ent_end, name=entity.name, id=entity.id))
            ent_i += 1

        # Update the start position for the next sentence
        start += len(sentences[sent_i]) + 1  # +1 accounts for the space or punctuation between sentences
        sent_i += 1

        # Yield the entities corresponding to the current sentence
        yield sent_entities


def convert_brat_relations(relations: Iterable[BratRelation]) -> Generator[Relation]:
    for brat_relation in relations:
        src = str(tuple(brat_relation.arg1.spans))
        dst = str(tuple(brat_relation.arg2.spans))
        relation = brat_relation.relation.upper()

        if relation not in ['TEMPORALITE', 'CAUSE-CONSEQUENCE'] and 'INCERTAIN' not in relation:
            yield Relation(src=src, dst=dst, name=relation)


def split_relations(relations: Iterable[Relation], entities: Sequence[Sequence[Entity]]) -> list[list[Relation]]:
    """
    Splits relations into sentence-specific relationships by mapping entity IDs to their indices
    within the corresponding sentence's entities.

    :param relations: An iterable of `Relation`.
    :param entities: A sequence of sequences, where each inner sequence contains `Entity` objects
                     corresponding to entities in a sentence.

    :return: A list of lists. Each inner list corresponds to a sentence and contains `Relation` objects
             for that sentence.

    Example:
    >>> e1 = Entity(name="Entity1", start=0, end=1, id="E1")
    >>> e2 = Entity(name="Entity2", start=2, end=3, id="E2")
    >>> e3 = Entity(name="Entity3", start=4, end=5, id="E3")
    >>> e4 = Entity(name="Entity4", start=6, end=7, id="E4")
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


def get_sentence_from_disk(path: Path) -> Generator[AnnotatedSentence]:
    dataset: BratDataset = BratDataset.from_directory(path.absolute())
    brat_file: BratFile

    for brat_file in (pbar := tqdm(dataset, total=len(dataset.brat_files))):
        file_path = Path(brat_file.txt_path)
        pbar.set_description(f'Load {file_path.name}')

        yield from convert_brat_file(brat_file)


def _get_trees(
    sentences: Iterable[AnnotatedSentence], *, corenlp_url: str, language: str
) -> Generator[tuple[AnnotatedSentence, ParentedTree], None, None]:
    """
    Parses a collection of sentences into syntax trees using CoreNLP.

    This function takes an iterable of sentences and processes each sentence through a CoreNLP server to obtain its syntax tree.
    The tree is then converted into a `ParentedTree` from NLTK.

    :param sentences: An iterable collection of `AnnotatedSentence` objects.
    :param corenlp_url: The URL of the CoreNLP server.
    :param language: The language to use for parsing.

    :yields: A tuple of the original `AnnotatedSentence` and its corresponding `ParentedTree`.

    Example:
        .. code-block:: python

            for sentence, tree in get_trees(sentences, corenlp_url="http://localhost:9000"):
                print(sentence, tree)

    """
    nltk_parser = CoreNLPParser(url=corenlp_url)
    properties = {
        'tokenize.language': language,
        'ssplit.eolonly': 'true',
    }

    for sentence in sentences:
        try:
            # Parse the sentence text using the CoreNLP server configured for French
            for rooted_tree in nltk_parser.parse_text(sentence.txt, properties=properties):
                # Each rooted_tree may contain multiple sentence subtrees; iterate over each
                for sent_tree in rooted_tree:
                    # Convert each subtree into a ParentedTree and yield along with the original sentence
                    yield sentence, ParentedTree.convert(sent_tree)

        except requests.exceptions.ConnectionError as error:
            # Handle connection issues with the CoreNLP server
            print(f'Cannot parse the following text due to {error.strerror} : "{sentence.txt}"')


def get_enriched_forest(
    sentences: Iterable[AnnotatedSentence], *, corenlp_url: str, language: str = 'French'
) -> Generator[ParentedTree, None, None]:
    """
    Enriches and processes syntax trees for a given collection of sentences.

    This function takes an iterable of sentences, parses each one into a syntax tree, enriches the tree by
    fixing coordination structures, adding extra information (entities and relations), and applying reductions.

    :param sentences: An iterable collection of `AnnotatedSentence`.
    :param corenlp_url: The URL of the CoreNLP server.
    :param language: The language to use for parsing.

    :yields: An enriched `ParentedTree` object for each sentence.

    Example:
        .. code-block:: python

            for enriched_tree in get_enriched_forest(sentences, corenlp_url="http://localhost:9000"):
                print(enriched_tree)

    """
    # Iterate over the parsed trees and associated sentences
    for sentence, tree in _get_trees(sentences, corenlp_url=corenlp_url, language=language):
        # Replace specific parenthesis tokens ('-LRB-' for '(', '-RRB-' for ')') in the leaf nodes
        for subtree in tree.subtrees(lambda x: x.height() == 2 and len(x) == 1 and x[0] in {'-LRB-', '-RRB-'}):
            subtree[0] = '(' if subtree[0] == '-LRB-' else ')'

        # Flatten the coordination in the tree structure
        fix_all_coord(tree)

        # Enrich the tree with named entities and relations from the sentence
        enrich_tree(tree, sentence.txt, sentence.entities, sentence.rels)

        # Reduce the tree structure removing unneeded nodes
        reduce_all(tree, set(NodeType))

        yield tree

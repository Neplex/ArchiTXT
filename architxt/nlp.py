from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

from bratlib.data import BratDataset, BratFile
from bratlib.data.annotation_types import Entity as BratEntity
from bratlib.data.annotation_types import Relation as BratRelation
from joblib import Parallel, delayed
from nltk.parse.corenlp import CoreNLPParser
from tqdm import tqdm
from unidecode import unidecode

from architxt.model import AnnotatedSentence, Entity, NodeType, Relation
from architxt.tree import ParentedTree, fix_all_coord, ins_ent_list, reduce_all

_parallel = Parallel(n_jobs=-2, require='sharedmem', return_as='generator')


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


def split_entities(entities: Iterable[Entity], sentences: Sequence[str]) -> Generator[list[Entity]]:
    entities = sorted(entities)
    ent_i = 0
    sent_i = 0

    start = 0
    while sent_i < len(sentences):
        sent_entities = []
        end = start + len(sentences[sent_i])

        while ent_i < len(entities) and entities[ent_i].end <= end:
            entity = entities[ent_i]
            ent_start = max(entity.start - start, 0)
            ent_end = min(entity.end - start, len(sentences[sent_i]))

            sent_entities.append(Entity(start=ent_start, end=ent_end, name=entity.name, id=entity.id))
            ent_i += 1

        start += len(sentences[sent_i]) + 1
        sent_i += 1
        yield sent_entities


def convert_brat_relations(relations: Iterable[BratRelation]) -> Generator[Relation]:
    for brat_relation in relations:
        src = tuple(brat_relation.arg1.spans)
        dst = tuple(brat_relation.arg2.spans)
        relation = brat_relation.relation.upper()

        if relation not in ['TEMPORALITE', 'CAUSE-CONSEQUENCE'] and 'INCERTAIN' not in relation:
            yield src, dst, relation


def split_relations(relations: Iterable[Relation], entities: Sequence[Sequence[Entity]]):
    relationship = []
    for _ in range(len(entities)):
        relationship.append([])

    for rel in relations:
        sent_i = None
        src_i = None
        dst_i = None
        for i in range(len(entities)):
            src_i = None
            dst_i = None

            for j in range(len(entities[i])):
                if entities[i][j].id == rel[0]:
                    src_i = j
                if entities[i][j].id == rel[1]:
                    dst_i = j

            if src_i is not None and dst_i is not None:
                sent_i = i
                break

        if sent_i:
            relationship[sent_i].append((src_i, dst_i, rel[2]))

    return relationship


def get_sentence_from_disk(path: Path) -> Generator[AnnotatedSentence]:
    dataset: BratDataset = BratDataset.from_directory(path.absolute())
    brat_file: BratFile

    for brat_file in (pbar := tqdm(dataset, total=len(dataset.brat_files))):
        file_path = Path(brat_file.txt_path)
        pbar.set_description(f'Load {file_path.name}')

        yield from convert_brat_file(brat_file)


def get_sentence_tree(sentence: AnnotatedSentence, *, parser: CoreNLPParser) -> ParentedTree:
    tree = ParentedTree(
        'ROOT',
        children=[
            ParentedTree.convert(sent_tree)
            for tree in parser.raw_parse_sents([sentence.txt], properties={'tokenize.language': 'French'})
            for rooted_tree in tree
            for sent_tree in rooted_tree
        ],
    )

    for subtree in tree.subtrees(lambda x: x.height() == 2 and len(x) == 1 and x[0] in {'-LRB-', '-RRB-'}):
        subtree[0] = '(' if subtree[0] == '-LRB-' else ')'

    return tree


def get_annotated_sentence_tree(sentence: AnnotatedSentence, *, parser: CoreNLPParser) -> ParentedTree | None:
    try:
        tree = get_sentence_tree(sentence, parser=parser)

        fix_all_coord(tree)
        ins_ent_list(tree, sentence.txt, sentence.entities, sentence.rels)
        reduce_all(tree, set(NodeType))

        return tree

    except Exception:
        return None


def get_annotated_rooted_forest(sentences: Iterable[AnnotatedSentence], *, url: str) -> ParentedTree:
    nltk_parser = CoreNLPParser(url=url)

    annotated_trees = _parallel(
        delayed(get_annotated_sentence_tree)(sentence, parser=nltk_parser) for sentence in tqdm(sentences)
    )
    annotated_trees = filter(lambda x: x is not None, annotated_trees)
    annotated_forest = ParentedTree('ROOT', children=list(annotated_trees))

    reduce_all(annotated_forest, set(NodeType))

    return annotated_forest

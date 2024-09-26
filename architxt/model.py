from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from os.path import commonprefix
from nltk.tokenize.util import align_tokens

TREE_POS = tuple[int, ...]


class NodeType(str, Enum):
    ENT = 'ENT'
    GROUP = 'GROUP'
    REL = 'REL'
    COLL = 'COLL'


class NodeLabel(str):
    type: NodeType
    name: str

    def __new__(cls, label_type: NodeType, label: str = ''):
        return super().__new__(cls, f'{label} ({label_type})')

    def __init__(self, label_type: NodeType, label: str = ''):
        self.name = str(label)
        self.type = label_type


@dataclass
class Entity:
    name: str
    start: int
    end: int
    id: str

    def token_index(self, sentence: str, tokens: list[str]) -> Generator[int]:
        token_spans = align_tokens(tokens, sentence)

        for i, token in enumerate(token_spans):
            if self.start <= token[1] and token[0] < self.end:
                yield i

    def __len__(self):
        return self.end - self.start

    def __lt__(self, other):
        return self.start < other.start


@dataclass
class TreeEntity:
    name: str
    positions: list[TREE_POS, ...]

    @property
    def root_pos(self) -> TREE_POS:
        prefix = commonprefix(self.positions)
        return prefix if prefix != self.positions[0] else prefix[:-1]

    def __len__(self):
        return len(self.positions)


class Relation(tuple[Entity, Entity]):
    name: str


@dataclass
class TreeRel:
    pos_start: TREE_POS
    pos_end: TREE_POS
    name: str


@dataclass
class AnnotatedSentence:
    txt: str
    entities: list[Entity]
    rels: list[Relation]

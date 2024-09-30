from dataclasses import dataclass
from enum import Enum
from os.path import commonprefix

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

    def __str__(self):
        return f'{self.type.value}::{self.name}' if self.name else self.type.value


@dataclass
class Entity:
    name: str
    start: int
    end: int
    id: str

    def __post_init__(self):
        if self.start < 0:
            raise ValueError("Start cannot be negative.")
        if self.start >= self.end:
            raise ValueError("Start cannot be larger than end.")

    def __len__(self):
        return self.end - self.start

    def __lt__(self, other):
        return self.start < other.start


@dataclass
class TreeEntity:
    name: str
    positions: list[TREE_POS]

    @property
    def root_pos(self) -> TREE_POS:
        prefix = commonprefix(self.positions)
        return prefix if prefix != self.positions[0] else prefix[:-1]

    def __post_init__(self):
        if not self.positions:
            raise ValueError("Cannot have empty list of positions.")

    def __len__(self):
        return len(self.positions)


@dataclass
class Relation:
    src: str  # Ent id
    dst: str  # Ent id
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

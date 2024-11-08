from dataclasses import dataclass
from enum import Enum
from os.path import commonprefix

TREE_POS = int | tuple[int, ...]


class NodeType(str, Enum):
    ENT = 'ENT'
    GROUP = 'GROUP'
    REL = 'REL'
    COLL = 'COLL'


class NodeLabel(str):
    type: NodeType
    name: str

    __slots__ = ('type', 'name')

    def __new__(cls, label_type: NodeType, label: str = ''):
        return super().__new__(cls, f'{label_type.value}::{label}' if label else label_type.value)

    def __init__(self, label_type: NodeType, label: str = ''):
        self.name = label
        self.type = label_type

    def __reduce__(self):
        return NodeLabel, (self.type, self.name)


@dataclass(slots=True)
class Entity:
    """A named entity"""

    name: str
    start: int
    end: int
    id: str

    def __post_init__(self):
        if self.start < 0:
            msg = "Start cannot be negative."
            raise ValueError(msg)

        if self.start >= self.end:
            msg = "Start cannot be larger than end."
            raise ValueError(msg)

    def __len__(self):
        return self.end - self.start

    def __lt__(self, other):
        return self.start < other.start


@dataclass(slots=True)
class TreeEntity:
    """An entity in a tree, the name is associate with a list of leaf tree position."""

    name: str
    positions: list[TREE_POS]

    @property
    def root_pos(self) -> tuple[int, ...]:
        """Get the position that cover every position of the entity."""
        prefix = commonprefix(self.positions)
        return prefix if prefix != self.positions[0] else prefix[:-1]

    def __post_init__(self):
        if not self.positions:
            msg = "Cannot have empty list of positions."
            raise ValueError(msg)

    def __len__(self):
        return len(self.positions)


@dataclass(slots=True)
class Relation:
    """A relation between two entities."""

    src: str  # Ent id
    dst: str  # Ent id
    name: str


@dataclass(slots=True)
class TreeRel:
    """A relation between two entities in a tree."""

    pos_start: TREE_POS
    pos_end: TREE_POS
    name: str


@dataclass(slots=True)
class AnnotatedSentence:
    """A sentence with Entity/Relation annotations."""

    txt: str
    entities: list[Entity]
    rels: list[Relation]

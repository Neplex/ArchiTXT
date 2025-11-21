from __future__ import annotations

from dataclasses import dataclass
from os.path import commonprefix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architxt.tree import TreePosition

__all__ = ['AnnotatedSentence', 'Entity', 'Relation', 'TreeEntity', 'TreeRel']


@dataclass(slots=True)
class Entity:
    """A named entity."""

    name: str
    start: int
    end: int
    id: str
    value: str

    def __post_init__(self) -> None:
        if self.start < 0:
            msg = "Start cannot be negative."
            raise ValueError(msg)

        if self.start >= self.end:
            msg = "Start cannot be larger than end."
            raise ValueError(msg)

    def __len__(self) -> int:
        return self.end - self.start

    def __lt__(self, other: Entity) -> bool:
        return self.start < other.start


@dataclass(slots=True)
class TreeEntity:
    """An entity in a tree, the name is associate with a list of leaf tree position."""

    name: str
    positions: list[TreePosition]
    value: str | None = None

    @property
    def root_pos(self) -> tuple[int, ...]:
        """Get the position that covers every position of the entity."""
        prefix = commonprefix(self.positions)
        return tuple(prefix) if prefix != self.positions[0] else tuple(prefix[:-1])

    def __post_init__(self) -> None:
        if not self.positions:
            msg = "Cannot have empty list of positions."
            raise ValueError(msg)

    def __len__(self) -> int:
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

    pos_start: TreePosition
    pos_end: TreePosition
    name: str


@dataclass(slots=True)
class AnnotatedSentence:
    """A sentence with Entity/Relation annotations."""

    txt: str
    entities: list[Entity]
    rels: list[Relation]

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from architxt.tree import Tree

__all__ = ['ForestInspector']


class ForestInspector:
    def __init__(self) -> None:
        self.total_trees = 0
        self.total_entities = 0
        self.total_nodes = 0
        self.sum_children = 0
        self.max_children = 0
        self.sum_height = 0
        self.max_height = 0
        self.sum_size = 0
        self.max_size = 0
        self.entity_count = Counter[str]()
        self.largest_tree: Tree | None = None

    @property
    def avg_height(self) -> float:
        """Get the average height of all trees."""
        return self.sum_height / self.total_trees if self.total_trees else 0

    @property
    def avg_size(self) -> float:
        """Get the average size (number of leaves) of all trees."""
        return self.sum_size / self.total_trees if self.total_trees else 0

    @property
    def avg_branching(self) -> float:
        """Get the average branching factor (children per node) across all trees."""
        return self.sum_children / self.total_nodes if self.total_nodes else 0

    def __call__(self, forest: Iterable[Tree]) -> Generator[Tree, None, None]:
        for tree in forest:
            self.total_trees += 1

            # Count and track heights
            height = tree.height
            self.sum_height += height
            if height > self.max_height:
                self.max_height = height
                self.largest_tree = tree

            # Count and track sizes
            size = len(tree.leaves())
            self.sum_size += size
            if size > self.max_size:
                self.max_size = size

            # Count entities
            entities = [ent.label for ent in tree.entities()]
            self.total_entities += len(entities)
            self.entity_count.update(entities)

            # Calculate branching factor
            for node in tree.subtrees():
                nb_children = len(node)
                self.total_nodes += 1
                self.sum_children += nb_children
                if nb_children > self.max_children:
                    self.max_children = nb_children

            yield tree

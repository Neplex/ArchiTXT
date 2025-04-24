from collections import Counter
from collections.abc import Generator, Iterable

from architxt.tree import Tree

__all__ = ['ForestInspector']


class ForestInspector:
    def __init__(self) -> None:
        self.total_trees = 0
        self.total_entities = 0
        self.sum_height = 0
        self.max_height = 0
        self.sum_size = 0
        self.max_size = 0
        self.entity_count = Counter()
        self.largest_tree = None

    @property
    def avg_height(self) -> float:
        return self.sum_height / self.total_trees if self.total_trees else 0

    @property
    def avg_size(self) -> float:
        return self.sum_size / self.total_trees if self.total_trees else 0

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
            entities = [ent.label.name for ent in tree.entities()]
            self.total_entities += len(entities)
            self.entity_count.update(entities)

            yield tree

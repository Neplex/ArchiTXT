from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from architxt.tree import NodeType, Tree, has_type

from .operation import Operation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from architxt.similarity import TREE_CLUSTER
    from architxt.tree import _SubTree

__all__ = [
    'ReduceBottomOperation',
    'ReduceTopOperation',
]


class ReduceOperation(Operation, ABC):
    """
    Base class for reduction operations.

    This class defines custom behavior for identifying subtrees to be reduced and applying the reduction operation.
    """

    @abstractmethod
    def subtrees_to_reduce(self, tree: Tree) -> Iterable[_SubTree]: ...

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:  # noqa: ARG002
        reduced = False

        for subtree in self.subtrees_to_reduce(tree):
            parent = subtree.parent
            position = subtree.position
            label = subtree.label
            old_labels = tuple(str(child.label) for child in parent)

            # Convert subtree's children into independent nodes
            new_children = (child.detach() for child in subtree[:])

            # Put children in the parent at the original subtree position
            parent_pos = subtree.parent_index
            parent[parent_pos : parent_pos + 1] = new_children

            new_labels = tuple(str(child.label) for child in parent)
            self._log_to_mlflow(
                {
                    'label': str(label),
                    'position': position,
                    'labels.old': old_labels,
                    'labels.new': new_labels,
                }
            )

            reduced = True

        return reduced


class ReduceBottomOperation(ReduceOperation):
    """
    Reduces the unlabelled nodes of a tree at the bottom-level.

    This function identifies subtrees that do not have a specific type but contain only children of type `ENT`.
    It then repositions these subtrees children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.
    """

    def subtrees_to_reduce(self, tree: Tree) -> Iterable[_SubTree]:
        return [
            subtree
            for subtree in tree.subtrees(include_self=False, reverse=True)
            if not has_type(subtree) and all(has_type(child, NodeType.ENT) for child in subtree)
        ]


class ReduceTopOperation(ReduceOperation):
    """
    Reduces the unlabelled nodes of a tree at the top-level.

    It identifies subtrees that do not have a specific type and repositions these subtrees children
    directly under their parent nodes, effectively "flattening" the tree structure at this level.
    """

    def subtrees_to_reduce(self, tree: Tree) -> Iterable[_SubTree]:
        return [subtree for subtree in tree if isinstance(subtree, Tree) and not has_type(subtree)]

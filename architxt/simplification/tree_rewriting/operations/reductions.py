from abc import ABC, abstractmethod
from collections.abc import Iterable

from architxt.similarity import TREE_CLUSTER
from architxt.tree import Tree, has_type

from .operation import Operation

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
    def subtrees_to_reduce(self, tree: Tree) -> Iterable[Tree]: ...

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:  # noqa: ARG002
        reduced = False

        # Iterate through subtrees in reverse order to ensure bottom-up processing
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

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.
    """

    def subtrees_to_reduce(self, tree: Tree) -> Iterable[Tree]:
        yield from tree.subtrees(lambda x: x.parent and x.has_entity_child() and not has_type(x))


class ReduceTopOperation(ReduceOperation):
    """
    Reduces the unlabelled nodes of a tree at the top-level.

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.
    """

    def subtrees_to_reduce(self, tree: Tree) -> Iterable[Tree]:
        for subtree in list(tree):
            if isinstance(subtree, Tree) and not has_type(subtree):
                yield subtree

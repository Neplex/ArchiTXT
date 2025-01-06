from copy import deepcopy

from architxt.similarity import TREE_CLUSTER
from architxt.tree import has_type

from .operation import Operation

__all__ = [
    'ReduceBottomOperation',
    'ReduceTopOperation',
]


class ReduceBottomOperation(Operation):
    """
    Reduces the unlabelled nodes of a tree at the bottom-level

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.
    """

    def apply(self, tree, *, equiv_subtrees: TREE_CLUSTER):
        reduced = False

        # Iterate through subtrees in reverse order to ensure bottom-up processing
        for subtree in list(tree.subtrees(lambda x: x.parent() and x.has_entity_child() and not has_type(x))):
            parent = subtree.parent()
            position = subtree.treeposition()
            label = subtree.label()
            old_labels = tuple(child.label() for child in parent)

            # Convert subtree's children into independent nodes
            new_children = [deepcopy(child) for child in subtree]

            # Put children in the parent at the original subtree position
            parent_pos = subtree.parent_index()
            parent[parent_pos : parent_pos + 1] = new_children

            new_labels = tuple(child.label() for child in parent)
            self._log_to_mlflow(
                {
                    'label': str(label),
                    'position': position,
                    'labels.old': old_labels,
                    'labels.new': new_labels,
                }
            )

            reduced = True

        return tree, reduced


class ReduceTopOperation(Operation):
    """
    Reduces the unlabelled nodes of a tree at the top-level

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.
    """

    def apply(self, tree, *, equiv_subtrees: TREE_CLUSTER):
        reduced = False

        for subtree in list(tree):
            if has_type(subtree):
                continue

            parent = subtree.parent()
            position = subtree.treeposition()
            label = subtree.label()
            old_labels = tuple(child.label() for child in parent)

            # Convert subtree's children into independent nodes
            new_children = [deepcopy(child) for child in subtree]

            # Put children in the parent at the original subtree position
            parent_pos = subtree.parent_index()
            parent[parent_pos : parent_pos + 1] = new_children

            new_labels = tuple(child.label() for child in parent)
            self._log_to_mlflow(
                {
                    'label': str(label),
                    'position': position,
                    'labels.old': old_labels,
                    'labels.new': new_labels,
                }
            )

            reduced = True

        return tree, reduced

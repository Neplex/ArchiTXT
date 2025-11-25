from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import more_itertools

from architxt.tree import NodeLabel, NodeType, Tree, has_type

from .operation import Operation

if TYPE_CHECKING:
    from architxt.similarity import TREE_CLUSTER

__all__ = [
    'FindRelationsOperation',
]


def _is_valid_relation(tree: Tree) -> bool:
    return len(tree) == 2 and has_type(tree[0], NodeType.GROUP) and has_type(tree[1], NodeType.GROUP)


class FindRelationsOperation(Operation):
    """
    Identifies and establishes hierarchical relationships between `GROUP` nodes within a tree structure.

    The function scans for subtrees that contain at least two distinct elements.
    When a `GROUP` node is found to have a relationship with a collection, that relationship
    is distributed between the `GROUP` node itself and each member of the collection.

    The operation can operate in two modes:
    1. Naming-only mode: Simply assigns labels to valid relations without altering the tree's structure.
    2. Structural modification mode: restructures the tree by creating relation nodes between groups and collections.
    """

    def __init__(self, *args: Any, naming_only: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.naming_only = naming_only

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:  # noqa: ARG002
        simplified = False

        # Traverse subtrees, starting with the deepest, containing exactly 2 children
        for subtree in sorted(
            tree.subtrees(
                lambda x: len(x) == 2
                and not has_type(x, {NodeType.ENT, NodeType.GROUP})
                and all(has_type(y, {NodeType.GROUP, NodeType.COLL}) for y in x)
            ),
            key=lambda x: x.depth,
            reverse=True,
        ):
            if _is_valid_relation(subtree):  # Group <-> Group
                if self._create_group_group_relation(subtree):
                    simplified = True

            elif not self.naming_only and self._create_group_collection_relation(subtree):  # Group <-> Collection
                simplified = True

        return simplified

    def _create_group_group_relation(self, tree: Tree) -> bool:
        modified = False

        if not _is_valid_relation(tree):
            return False

        if tree[0].label.name == tree[1].label.name:
            return False

        # Create and set the relationship label
        label = sorted([tree[0].label.name, tree[1].label.name])
        rel_label = NodeLabel(NodeType.REL, f'{label[0]}<->{label[1]}')

        # Log relation creation in MLFlow, if active
        if not has_type(tree, NodeType.REL):
            modified = True
            self._log_to_mlflow({'name': rel_label})

        tree.label = rel_label
        return modified

    def _create_group_collection_relation(self, tree: Tree) -> bool:
        if len(tree) != 2:
            return False

        if has_type(tree[0], NodeType.GROUP) and has_type(tree[1], NodeType.COLL):
            group, collection = tree[0], tree[1]
        elif has_type(tree[0], NodeType.COLL) and has_type(tree[1], NodeType.GROUP):
            collection, group = tree[0], tree[1]
        else:
            return False

        # If a valid Group-Collection pair is found, create relationships for each
        if (
            len(collection) == 0
            or not all(has_type(x, NodeType.GROUP) for x in collection)
            or not more_itertools.all_equal(collection, lambda x: x.label.name)
        ):
            warnings.warn("Collection is empty or does not contain homogeneous GROUP nodes.")
            return False

        group_label = group.label.name
        collection_label = collection[0].label.name

        if group_label == collection_label:
            return False

        label1, label2 = sorted((group_label, collection_label))
        rel_label = NodeLabel(NodeType.REL, f'{label1}<->{label2}')

        # Create relationship nodes for each element in the collection
        for coll_group in collection[:]:
            rel_tree = Tree(rel_label, children=[group.copy(), coll_group.detach()])
            tree.append(rel_tree)  # Add new relationship to subtree

            # Log relation creation in MLFlow, if active
            self._log_to_mlflow({'name': rel_label.name})

        tree.remove(group)
        tree.remove(collection)

        return True

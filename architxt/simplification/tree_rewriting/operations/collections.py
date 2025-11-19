from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import more_itertools

from architxt.tree import NodeLabel, NodeType, Tree, has_type

from .operation import Operation

if TYPE_CHECKING:
    from architxt.similarity import TREE_CLUSTER
    from architxt.tree import _TypedSubTree

__all__ = [
    'FindCollectionsOperation',
]


class FindCollectionsOperation(Operation):
    """
    Identifies and groups nodes into collections.

    The operation can operate in two modes:
    1. Naming-only mode: Simply assigns labels to valid collections without altering the tree's structure.
    2. Structural modification mode: Groups nodes into collection sets, updates their labels, and restructures
    the tree by creating collection nodes.
    """

    def __init__(self, *args: Any, naming_only: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.naming_only = naming_only

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:  # noqa: ARG002
        simplified = False

        for subtree in sorted(
            tree.subtrees(
                lambda x: not has_type(x, {NodeType.ENT, NodeType.GROUP, NodeType.REL})
                and any(has_type(y, {NodeType.GROUP, NodeType.REL, NodeType.COLL}) for y in x)
            ),
            key=lambda x: x.depth,
            reverse=True,
        ):
            if has_type(subtree, NodeType.COLL):  # Renaming only
                subtree.label = NodeLabel(NodeType.COLL, subtree[0].label.name)
                continue

            # Naming-only mode: apply labels without modifying the tree structure
            if self.naming_only:
                if has_type(subtree[0], {NodeType.GROUP, NodeType.REL}) and more_itertools.all_equal(
                    subtree, key=lambda x: x.label
                ):
                    subtree.label = NodeLabel(NodeType.COLL, subtree[0].label.name)
                    simplified = True
                continue

            # Group nodes by shared label and organize them into collection sets for structural modification
            if self._merge_equivalent_siblings_into_collection(subtree):
                simplified = True

        return simplified

    def _merge_equivalent_siblings_into_collection(self, subtree: Tree) -> bool:
        """Find duplicate labels and merge them into real COLL nodes. Returns True if modified."""
        modified = False

        equiv_groups: defaultdict[str, list[_TypedSubTree]] = defaultdict(list)
        for child in subtree:
            if has_type(child, {NodeType.GROUP, NodeType.REL, NodeType.COLL}):
                equiv_groups[child.label.name].append(child)

        # Only groups with duplicates
        duplicate_groups = (
            sorted(trees, key=lambda t: t.parent_index) for trees in equiv_groups.values() if len(trees) > 1
        )

        for coll_set in duplicate_groups:
            index = coll_set[0].parent_index
            label = NodeLabel(NodeType.COLL, coll_set[0].label.name)

            # Prepare a new collection of nodes (merging if some nodes are already collections)
            children: list[Tree] = []
            for node in coll_set:
                if has_type(node, NodeType.COLL):
                    children.extend(child.detach() for child in node[:])
                    node.detach()
                else:
                    children.append(node.detach())

            self._log_to_mlflow({'name': label.name, 'size': len(children)})
            modified = True

            if len(subtree) == 0:
                # If the entire subtree is a single collection, reuse it
                subtree.label = label
                subtree[:] = children
            else:
                # Insert the new collection node at the appropriate index
                coll_tree = Tree(label, children=children)
                subtree.insert(index, coll_tree)

        return modified

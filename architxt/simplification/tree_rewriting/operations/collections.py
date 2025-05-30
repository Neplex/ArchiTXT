from itertools import groupby
from typing import Any

import more_itertools

from architxt.similarity import TREE_CLUSTER
from architxt.tree import NodeLabel, NodeType, Tree, has_type

from .operation import Operation

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
            for coll_tree_set in sorted(
                filter(
                    lambda x: len(x) > 1,
                    (
                        sorted(equiv_set, key=lambda x: x.parent_index)
                        for _, equiv_set in groupby(
                            sorted(
                                filter(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL, NodeType.COLL}), subtree),
                                key=lambda x: x.label.name,
                            ),
                            key=lambda x: x.label.name,
                        )
                    ),
                ),
                key=lambda x: x[0].parent_index,
            ):
                index = coll_tree_set[0].parent_index
                label = NodeLabel(NodeType.COLL, coll_tree_set[0].label.name)

                # Prepare a new collection of nodes (merging if some nodes are already collections)
                children: list[Tree] = []
                for coll_tree in coll_tree_set:
                    if has_type(coll_tree, NodeType.COLL):
                        # Merge collection elements
                        children.extend(child.detach() for child in coll_tree[:])
                        coll_tree.detach()

                    else:
                        children.append(coll_tree.detach())

                # Log the creation of a new collection in MLFlow, if active
                self._log_to_mlflow(
                    {
                        'name': label.name,
                        'size': len(children),
                    }
                )
                simplified = True

                # If the entire subtree is a single collection, update its label and structure directly
                if len(subtree) == 0:
                    subtree.label = label
                    subtree[:] = children

                else:
                    # Insert the new collection node at the appropriate index
                    coll_tree = Tree(label, children=children)
                    subtree.insert(index, coll_tree)

        return simplified

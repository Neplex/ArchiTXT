from copy import deepcopy
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

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> tuple[Tree, bool]:  # noqa: ARG002
        simplified = False

        for subtree in sorted(
            tree.subtrees(
                lambda x: not has_type(x) and any(has_type(y, {NodeType.GROUP, NodeType.REL, NodeType.COLL}) for y in x)
            ),
            key=lambda x: x.depth,
            reverse=True,
        ):
            # Naming-only mode: apply labels without modifying tree structure
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
                # Prepare a new collection of nodes (merging if some nodes are already collections)
                coll_elements = []
                for coll_tree in coll_tree_set:
                    if has_type(coll_tree, NodeType.COLL):
                        coll_elements.extend(coll_tree)  # Merge collection elements
                    else:
                        coll_elements.append(coll_tree)

                # Prepare the collection node
                label = NodeLabel(NodeType.COLL, coll_tree_set[0].label.name)
                children = [deepcopy(tree) for tree in coll_elements]

                # Log the creation of a new collection in MLFlow, if active
                self._log_to_mlflow(
                    {
                        'name': label.name,
                        'size': len(children),
                    }
                )
                simplified = True

                # If the entire subtree is a single collection, update its label and structure directly
                if len(subtree) == len(coll_tree_set):
                    subtree.label = label
                    subtree.clear()
                    subtree.extend(children)

                else:
                    index = coll_tree_set[0].parent_index

                    # Remove nodes of the current collection set from the subtree
                    for coll_tree in coll_tree_set:
                        subtree.pop(coll_tree.parent_index, recursive=False)

                    # Insert the new collection node at the appropriate index
                    coll_tree = Tree(label, children=children)
                    subtree.insert(index, coll_tree)

        return tree, simplified

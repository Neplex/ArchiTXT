from typing import Any

from architxt.similarity import TREE_CLUSTER
from architxt.tree import NodeLabel, NodeType, Tree, has_type

from .operation import Operation

__all__ = [
    'FindRelationsOperation',
]


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

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:  # noqa: ARG002, C901
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
            if has_type(subtree, NodeType.REL):  # Renaming only
                label = sorted([subtree[0].label.name, subtree[1].label.name])
                subtree.label = NodeLabel(NodeType.REL, f'{label[0]}<->{label[1]}')
                continue

            group = None
            collection = None

            # Group <-> Group
            if has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.GROUP):
                if subtree[0].label.name == subtree[1].label.name:
                    continue

                # Create and set the relationship label
                label = sorted([subtree[0].label.name, subtree[1].label.name])
                subtree.label = NodeLabel(NodeType.REL, f'{label[0]}<->{label[1]}')

                # Log relation creation in MLFlow, if active
                simplified = True
                self._log_to_mlflow(
                    {
                        'name': f'{label[0]}<->{label[1]}',
                    }
                )
                continue

            # If only naming relationships, skip further processing
            if self.naming_only:
                continue

            # Group <-> Collection
            if has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.COLL):
                group, collection = subtree[0], subtree[1]

            elif has_type(subtree[0], NodeType.COLL) and has_type(subtree[1], NodeType.GROUP):
                collection, group = subtree[0], subtree[1]

            # If a valid Group-Collection pair is found, create relationships for each
            if group and collection and has_type(collection[0], NodeType.GROUP):
                if collection[0].label == group.label:
                    continue

                # Create relationship nodes for each element in the collection
                for coll_group in collection[:]:
                    label1, label2 = sorted([group.label.name, coll_group.label.name])
                    rel_label = NodeLabel(NodeType.REL, f'{label1}<->{label2}')
                    rel_tree = Tree(rel_label, children=[group.copy(), coll_group.detach()])
                    subtree.append(rel_tree)  # Add new relationship to subtree

                    # Log relation creation in MLFlow, if active
                    simplified = True
                    self._log_to_mlflow(
                        {
                            'name': rel_label.name,
                        }
                    )

                subtree.remove(group)
                subtree.remove(collection)

        return simplified

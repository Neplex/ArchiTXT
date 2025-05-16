from collections.abc import Iterable
from contextlib import nullcontext

import more_itertools
from tqdm.auto import tqdm

from architxt.bucket import TreeBucket
from architxt.tree import NodeLabel, NodeType, Tree, has_type
from architxt.utils import BATCH_SIZE

__all__ = ['simple_rewrite']


def _simple_rewrite_tree(tree: Tree, group_ids: dict[tuple[str, ...], str]) -> None:
    """Rewrite of a single tree."""
    if has_type(tree, NodeType.ENT) or not tree.has_unlabelled_nodes():
        return

    entities = tree.entity_labels()
    group_key = tuple(sorted(entities))

    if group_key not in group_ids:
        group_ids[group_key] = str(len(group_ids) + 1)

    group_label = NodeLabel(NodeType.GROUP, group_ids[group_key])
    group_entities: list[Tree] = []

    for entity in tree.entities():
        if entity.label.name in entities:
            group_entities.append(entity.copy())
            entities.remove(entity.label.name)

    group_tree = Tree(group_label, group_entities)
    tree[:] = [group_tree]


def simple_rewrite(forest: Iterable[Tree], *, commit: bool | int = BATCH_SIZE) -> None:
    """
    Rewrite a forest into a valid schema, treating each tree as a distinct group.

    This function processes each tree in the forest, collapsing its entities into a single
    group node if the tree contains unlabelled nodes.
    Each unique combination of entity labels is assigned a consistent group ID.
    Duplicate entities are removed.

    :param forest: A forest to be rewritten in place.
    :param commit: When working with a `TreeBucket`, changes can be committed automatically .
        - If False, no commits are made. Use this for small forests where you want to commit manually later.
        - If True, commits after processing the entire forest in one transaction.
        - If an integer, commits after processing every N tree.
        To avoid memory issues with large forests, we recommend using batch commit on large forests.
    """
    group_ids: dict[tuple[str, ...], str] = {}

    if commit and isinstance(forest, TreeBucket) and isinstance(commit, int):
        for chunk in more_itertools.ichunked(tqdm(forest, desc="Rewriting trees"), commit):
            with forest.transaction():
                for tree in chunk:
                    _simple_rewrite_tree(tree, group_ids)

    else:
        with forest.transaction() if commit and isinstance(forest, TreeBucket) else nullcontext():
            for tree in tqdm(forest, desc="Rewriting trees"):
                _simple_rewrite_tree(tree, group_ids)

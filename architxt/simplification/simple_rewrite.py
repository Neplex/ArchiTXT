from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import more_itertools
from tqdm.auto import tqdm

from architxt.bucket import TreeBucket
from architxt.tree import NodeLabel, NodeType, Tree, has_type
from architxt.utils import get_commit_batch_size

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def simple_rewrite(forest: Iterable[Tree], *, commit: bool | int = True) -> None:
    """
    Rewrite a forest into a valid schema, treating each tree as a distinct group.

    This function processes each tree in the forest, collapsing its entities into a single
    group node if the tree contains unlabelled nodes.
    Each unique combination of entity labels is assigned a consistent group ID.
    Duplicate entities are removed.

    :param forest: A forest to be rewritten in place.
    :param commit: Commit automatically if using TreeBucket. If already in a transaction not commit is applied.
        - If False, no commits are made, it relies on the current transaction.
        - If True (default), commits in batch.
        - If an integer, commits every N tree.
        To avoid memory issues, we recommend using incremental commit with large iterables.
    """
    batch_size = get_commit_batch_size(commit)
    group_ids: dict[tuple[str, ...], str] = {}
    trees = tqdm(forest, desc="Rewriting trees")

    for chunk in more_itertools.ichunked(trees, batch_size):
        with forest.transaction() if isinstance(forest, TreeBucket) and commit else nullcontext():
            for tree in chunk:
                _simple_rewrite_tree(tree, group_ids)

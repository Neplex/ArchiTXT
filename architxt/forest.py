from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anyio import open_file

from architxt.bucket import TreeBucket
from architxt.tree import Forest, Tree

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Generator, Iterable
    from pathlib import Path

__all__ = [
    'async_update_forest',
    'export_forest_to_jsonl',
    'export_forest_to_jsonl_async',
    'import_forest_from_jsonl',
    'update_forest',
]


def export_forest_to_jsonl(path: Path, forest: Iterable[Tree]) -> None:
    """
    Export a forest of :py:class:`~architxt.tree.Tree` objects to a JSONL file.

    :param path: Path to the output JSONL file.
    :param forest: Iterable of :py:class:`~architxt.tree.Tree` objects to export.
    """
    with path.open('w', encoding='utf-8') as f:
        for tree in forest:
            f.write(json.dumps(tree.to_json(), ensure_ascii=False) + '\n')


async def export_forest_to_jsonl_async(path: Path, forest: Iterable[Tree]) -> None:
    """
    Export a forest of :py:class:`~architxt.tree.Tree` objects to a JSONL file.

    :param path: Path to the output JSONL file.
    :param forest: Iterable of :py:class:`~architxt.tree.Tree` objects to export.
    """
    async with await open_file(path, 'w', encoding='utf-8') as f:
        for tree in forest:
            await f.write(json.dumps(tree.to_json(), ensure_ascii=False) + '\n')


def import_forest_from_jsonl(path: Path) -> Generator[Tree, None, None]:
    """
    Import a forest of :py:class:`~architxt.tree.Tree` objects from a JSONL file.

    :param path: Path to the input JSONL file.
    :yield: :py:class:`~architxt.tree.Tree` objects.
    """
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not (line := line.strip()):
                continue

            data = json.loads(line)
            yield Tree.from_json(data)


def update_forest(forest: Forest, trees: Iterable[Tree], *, commit: bool = False) -> None:
    """
    Update a forest with new trees.

    :param forest: The forest to update.
    :param trees: Iterable of :py:class:`~architxt.tree.Tree` objects to add to the forest.
    :param commit: Whether to commit the changes immediately (only relevant for database-backed forests).
    """
    if isinstance(forest, TreeBucket):
        forest.update(trees, commit=commit)
    elif isinstance(forest, list):
        forest[:] = [*trees]
    elif isinstance(forest, set):
        forest.update(trees)
    else:
        msg = f'Unsupported forest type: {type(forest)}'
        raise TypeError(msg)


async def async_update_forest(forest: Forest, trees: AsyncIterable[Tree], *, commit: bool = False) -> None:
    """
    Update a forest with new trees asynchronously.

    :param forest: The forest to update.
    :param trees: Iterable of :py:class:`~architxt.tree.Tree` objects to add to the forest.
    :param commit: Whether to commit the changes immediately (only relevant for database-backed forests).
    """
    if isinstance(forest, TreeBucket):
        await forest.async_update(trees, commit=commit)
    elif isinstance(forest, list):
        forest[:] = [tree async for tree in trees]
    elif isinstance(forest, set):
        forest.update({tree async for tree in trees})
    else:
        msg = f'Unsupported forest type: {type(forest)}'
        raise TypeError(msg)

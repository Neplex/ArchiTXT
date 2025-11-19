from __future__ import annotations

import json
from typing import TYPE_CHECKING

from architxt.tree import Tree

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path

__all__ = ['export_forest_to_jsonl', 'import_forest_from_jsonl']


def export_forest_to_jsonl(path: Path, forest: Iterable[Tree]) -> None:
    """
    Export a forest of :py:class:`~architxt.tree.Tree` objects to a JSONL file.

    :param path: Path to the output JSONL file.
    :param forest: Iterable of :py:class:`~architxt.tree.Tree` objects to export.
    """
    with path.open('w', encoding='utf-8') as f:
        for tree in forest:
            f.write(json.dumps(tree.to_json(), ensure_ascii=False) + '\n')


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

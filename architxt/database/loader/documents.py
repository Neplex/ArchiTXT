from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import more_itertools
import pandas as pd
import toml
import xmltodict
from ruamel import yaml

from architxt.tree import NodeLabel, NodeType, Tree, has_type

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from io import BytesIO

__all__ = ['read_document']

FILE_PARSERS: Sequence[Callable[[BytesIO | BinaryIO], dict[str, Any] | list[Any]]] = (
    json.load,  # JSON
    lambda x: toml.loads(x.read().decode()),  # TOML
    lambda x: list(yaml.YAML().load_all(x)),  # YAML
    lambda x: xmltodict.parse(x.read()),  # XML
    lambda x: pd.read_csv(x).to_dict(orient='records'),  # CSV
    lambda x: {k: df.to_dict(orient='records') for k, df in pd.read_excel(x, sheet_name=None).items()},  # EXCEL
)


def read_document(
    file: str | Path | BytesIO | BinaryIO,
    *,
    raw_read: bool = False,
    root_name: str = 'ROOT',
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Read the file as a data tree.

    XML is parsed according to https://www.xml.com/pub/a/2006/05/31/converting-between-xml-and-json.html

    :param file: The document file to read.
    :param raw_read: If enabled, the tree corresponds to the document without any transformation applied.
    :param root_name: The root node name.
    :param sample: Maximum number of samples to get for each collection.
        If 0, all samples are returned.
    :return: A list of trees representing the database.
    """
    raw_data = read_document_file(file)
    document_tree = read_tree(raw_data, root_name=root_name)

    if raw_read:
        yield document_tree
        return

    yield from parse_document_tree(document_tree, sample=sample)


def read_document_file(file: str | Path | BytesIO | BinaryIO) -> dict[str, Any] | list[Any]:
    """
    Read and parse a document file like XML, JSON, or CSV.

    :param file: The document database file to read.
    :return: The parsed contents of the file.

    :raises FileNotFoundError: If the file does not exist.
    :raises OSError: If the file cannot be read.
    :raises ValueError: If the file cannot be read or is empty.
    """
    should_close = False
    document_db: BytesIO | BinaryIO

    if isinstance(file, str | Path):
        document_db = Path(file).open('rb')  # noqa: SIM115
        should_close = True
    else:
        document_db = file

    try:
        data = parse_file(document_db)

    finally:
        if should_close:
            document_db.close()

    if not data:
        msg = 'Empty document'
        raise ValueError(msg)

    return data


def parse_file(file: BytesIO | BinaryIO) -> dict[str, Any] | list[Any]:
    """
    Parse a document database file like XML, JSON, or CSV.

    :param file: A file-like object opened for reading.
    :return: The parsed content of the file as a Python nested object.
    :raises: ValueError if none of the available parsers are able to process the input file.
    """
    cursor = file.tell()

    for parser in FILE_PARSERS:
        try:
            return parser(file)

        except Exception:  # noqa: PERF203
            file.seek(cursor)
            continue

    msg = 'Unsupported file type'
    raise ValueError(msg)


def read_tree(data: dict[str, Any] | list[Any], *, root_name: str = 'ROOT') -> Tree:
    """
    Recursively converts a document nested structure into a tree.

    - Dictionaries are treated as groups.
    - Lists are treated as collections.
    - Leaf elements are treated as entities.

    If a list contains only a single collection, the function flattens the output by returning
    that collection directly instead of nesting it under another collection node.

    :param data: The input data structure to be converted into a Tree.
    :param root_name: The label for the current node.
    :return: A nested tree structure corresponding to the input data.
    """
    root_name = root_name.replace(' ', '_').lower()

    sub_elements: Iterable[tuple[str, Any]] = (
        data.items() if isinstance(data, dict) else ((root_name, item) for item in data)
    )

    children = []
    for name, sub_element in sub_elements:
        if isinstance(sub_element, dict | list):  # Recursively process nested structures
            children.append(read_tree(sub_element, root_name=name))

        else:  # Leaf node becomes an entity
            ent_label = NodeLabel(NodeType.ENT, str(name).replace(' ', '_').lower())
            children.append(Tree(ent_label, [str(sub_element)]))

    # Flatten if the result is a single collection node
    if len(children) == 1 and has_type(children[0], NodeType.COLL):
        return children[0]

    label = NodeLabel(NodeType.COLL, root_name) if isinstance(data, list) else root_name
    return Tree(label, children)


def parse_document_tree(tree: Tree, *, sample: int = 0) -> Generator[Tree, None, None]:
    """
    Parse a document tree and yields processed subtrees based on collection grouping.

    - If the root node is **not** a collection, the entire tree is processed and a single result is yielded.
    - If the root node **is** a collection, each child subtree is individually processed and yielded.

    TODO: Enhance tree decomposition for nested collections.
        If no collection exists at the root level, consider splitting at the closest
        collection and duplicating the path to the root for each collection element.

    :param tree: The nested tree to be parsed.
    :param sample: Maximum number of samples to get for each collection.
        If 0, all samples are returned.
    :yield: Trees representing the database.
    """
    trees = tree if has_type(tree, NodeType.COLL) else [tree]

    for tree in trees:
        _group, parsed_tree = traverse_tree(tree, sample=sample)
        if len(parsed_tree):
            yield parsed_tree


def traverse_tree(tree: Tree, *, sample: int = 0) -> tuple[Tree, Tree]:
    """
    Recursively traverses and transforms a nested tree into a valid metamodel structure.

    The function extracts entity nodes and groups them under a single group node.
    It then establishes relations between this group and any nested subgroups.

    :param tree: The tree to traverse and transform.
    :param sample: Maximum number of samples to get for each collection.
        If 0, all samples are returned.
    :returns: A tuple containing:
        - The group to anchor too for parent relationship.
        - The transformed tree converting subgroup to relations.
    """
    if has_type(tree, NodeType.ENT):
        # Encapsulate entities into a group
        group_label = NodeLabel(NodeType.GROUP, tree.label.name)
        group_node = Tree(group_label, [tree.copy()])
        return group_node, group_node

    if has_type(tree, NodeType.COLL):
        children = more_itertools.sample(tree, sample) if sample else tree
        updated_children = [traverse_tree(child, sample=sample)[0] for child in children]
        updated_tree = Tree(tree.label, updated_children)
        return updated_tree, updated_tree

    # Separate entities and non-entities
    entities = [subtree.copy() for subtree in tree if has_type(subtree, NodeType.ENT)]
    non_entities = [subtree for subtree in tree if not has_type(subtree, NodeType.ENT)]

    # Group node for entities
    group_label = NodeLabel(NodeType.GROUP, tree.label)
    group_node = Tree(group_label, entities)

    relationship_nodes: list[Tree] = []

    for child in non_entities:
        child_group, child_tree = traverse_tree(child, sample=sample)

        if child_tree.label == 'ROOT':
            # extend relations recursively
            relationship_nodes.extend(grandchild.copy() for grandchild in child_tree)

        if has_type(child_group, NodeType.COLL):  # Create relationships with each element in the collection
            children = more_itertools.sample(child_group, sample) if sample else child_group
        else:
            children = [child_group]

        for element in children:
            element_label = element.label.name if isinstance(element.label, NodeLabel) else element.label
            rel_label = NodeLabel(NodeType.REL, f'{group_label.name}<->{element_label}')
            relationship_nodes.append(Tree(rel_label, [group_node.copy(), element.copy()]))

    # Return the group node and either a tree of relations or just the group if there are no relations
    return group_node, Tree('ROOT', relationship_nodes) if relationship_nodes else group_node

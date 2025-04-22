from collections.abc import Iterable

import neo4j

from architxt.tree import NodeType, Tree, has_type


def export_graph(
    forest: Iterable[Tree],
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the graph instance as a dictionary using Neo4j.

    :param session: Neo4j session.
    :param forest: The forest to export.
    :return: A generator that yields dictionaries representing the graph.
    """
    for tree in forest:
        export_tree(tree, session=session)


def export_tree(
    tree: Tree,
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the tree to the graph.

    :param tree: Tree to export.
    :param session: Neo4j session.
    """
    for group in tree.subtrees(lambda subtree: has_type(subtree, NodeType.GROUP)):
        export_group(group, session=session)

    for relation in tree.subtrees(lambda subtree: has_type(subtree, NodeType.REL)):
        export_relation(relation, session=session)


def export_relation(
    tree: Tree,
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the relation to the graph.

    :param tree: Relation to export.
    :param session: Neo4j session.
    """
    # Order is arbitrary, a better strategy could be used to determine source and target nodes
    src, dest = sorted(tree, key=lambda x: x.label)
    if tree.metadata.get('source') != src.label.name:
        src, dest = dest, src

    rel_name = tree.metadata.get('source_column', tree.label.replace('<->', '_'))

    session.run(f"""
    MATCH (src:`{src.label.name}` {{ {', '.join(f'`{k}`: {v!r}' for k, v in get_properties(src).items())} }})
    MATCH (dest:`{dest.label.name}` {{ {', '.join(f'`{k}`: {v!r}' for k, v in get_properties(dest).items())} }})
    MERGE (src)-[r:`{rel_name}`]->(dest)
    """)


def export_group(
    group: Tree,
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the group to the graph.

    :param group: Group to export.
    :param session: Neo4j session.
    """
    properties = get_properties(group)
    session.run(f"MERGE (n:`{group.label.name}` {{ {', '.join(f'`{k}`: {v!r}' for k, v in properties.items())} }})")


def get_properties(node: Tree) -> dict[str, str]:
    """
    Get the properties of a node.

    :param node: Node to get properties from.
    :return: Dictionary of properties.
    """
    return {child.label.name: child[0] for child in node if has_type(child, NodeType.ENT)}

import neo4j
from tqdm.auto import tqdm

from architxt.tree import Forest, NodeType, Tree


def export_graph(
    session: neo4j.Session,
    *,
    forest: Forest,
) -> None:
    """
    Export the graph instance as a dictionary using Neo4j.

    :param session: Neo4j session.
    :return: A generator that yields dictionaries representing the graph.
    """
    for tree in tqdm(forest, desc="Exporting graph"):
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
    for node in tree:
        if node.label().type == NodeType.GROUP:
            export_group(node, session=session)
        elif node.label().type == NodeType.REL:
            export_relation(node, session=session)


def export_relation(
    tree: Tree,
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the relation to the graph.

    :param relation: Relation to export.
    :param session: Neo4j session.
    """
    node1 = tree[0]
    node2 = tree[1]
    rel_name = tree[0].label().name + tree[1].label().name
    if tree.label().data:
        if tree.label().data['source'] == tree[1].label().name:
            node1, node2 = node2, node1
        if tree.label().data['source_column']:
            rel_name = tree.label().data['source_column']
    properties_group1 = get_properties(node1)
    properties_group2 = get_properties(node2)
    command = f"""
    MERGE (n1:{tree[0].label().name} {{ {', '.join(f'{k}: {v!r}' for k, v in properties_group1.items())} }})
    MERGE (n2:{tree[1].label().name} {{ {', '.join(f'{k}: {v!r}' for k, v in properties_group2.items())} }})
    MERGE (n1)-[r:{rel_name}]->(n2)
    """
    session.run(command)


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
    command = f"MERGE (n:{group.label().name} {{ {', '.join(f'{k}: {v!r}' for k, v in properties.items())} }})"
    session.run(command, group=group.label().name, **properties)


def get_properties(node: Tree) -> dict[str, str]:
    """
    Get the properties of a node.

    :param node: Node to get properties from.
    :return: Dictionary of properties.
    """
    properties = {}
    for child in node:
        if child.label().type == NodeType.ENT:
            properties[child.label().name] = child[0]
    return properties

from collections.abc import Iterable

import neo4j

from architxt.schema import Schema
from architxt.tree import NodeLabel, NodeType, Tree, has_type


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
    edge_data = get_edge_with_data(forest)
    for tree in forest:
        export_tree(tree, session=session, edge_data=edge_data)
    delete_id_column(session)


def get_edge_with_data(forest: Iterable[Tree]) -> set[str]:
    """
    Get the edges with data from the forest.

    :param forest: The forest to export.
    :return: A set of edges with data.
    """
    relations_with_data = set()
    schema = Schema.from_forest(forest)
    relation = schema.relations
    relation_count = {}

    for rel in relation.values():
        if rel[0] not in relation_count:
            relation_count[rel[0]] = 0
        if rel[1] not in relation_count:
            relation_count[rel[1]] = 0
        if rel[3]:
            relation_count[rel[1]] += 1
            relation_count[rel[0]] += 3
        else:
            relation_count[rel[0]] += 1
            relation_count[rel[1]] += 3

    for rel in relation_count:
        if relation_count[rel] == 2:
            relations_with_data.add(NodeLabel.fromstring(str(rel)))

    return relations_with_data


def export_tree(
    tree: Tree,
    *,
    session: neo4j.Session,
    edge_data: set[str],
) -> None:
    """
    Export the tree to the graph.

    :param tree: Tree to export.
    :param session: Neo4j session.
    """
    for group in tree.subtrees(lambda subtree: has_type(subtree, NodeType.GROUP)):
        if group.label not in edge_data:
            export_group(group, session=session)

    export_edge_data = {}
    for relation in tree.subtrees(lambda subtree: has_type(subtree, NodeType.REL)):
        if relation[0].label in edge_data:
            if relation[0] not in export_edge_data:
                export_edge_data[relation[0]] = set()
            export_edge_data[relation[0]].add(relation[1])
        elif relation[1].label in edge_data:
            if relation[1] not in export_edge_data:
                export_edge_data[relation[1]] = set()
            export_edge_data[relation[1]].add(relation[0])
        else:
            export_relation(relation, session=session)
    export_relation_edge_with_data(export_edge_data, session=session)


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
    MATCH (src:`{src.label.name}` {{_architxt_group_id: '{src.oid}'}})
    MATCH (dest:`{dest.label.name}` {{_architxt_group_id: '{dest.oid}'}})
    MERGE (src)-[r:`{rel_name}`]->(dest)
    """)


def export_relation_edge_with_data(
    edge_data: dict[Tree, set[Tree]],
    *,
    session: neo4j.Session,
) -> None:
    """
    Export the relation edge with data to the graph.

    :param edge_data: Dictionary of edges with data.
    :param session: Neo4j session.
    """
    for edge, relations in edge_data.items():
        relations = list(relations)
        src = relations[0]
        dest = relations[1]

        session.run(f"""
        MATCH (src:`{src.label.name}` {{_architxt_group_id: '{src.oid}'}})
        MATCH (dest:`{dest.label.name}` {{_architxt_group_id: '{dest.oid}'}})
        MERGE (src)-[r:{edge.label.name} {{ {', '.join(f'{k}: {v!r}' for k, v in get_properties(edge).items())} }}]->(dest)
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
    session.run(
        f"MERGE (n:`{group.label.name}` {{ _architxt_group_id: '{group.oid}', {', '.join(f'`{k}`: {v!r}' for k, v in properties.items())} }})"
    )
    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{group.label.name}`) ON (n._architxt_group_id)")


def get_properties(node: Tree) -> dict[str, str]:
    """
    Get the properties of a node.

    :param node: Node to get properties from.
    :return: Dictionary of properties.
    """
    return {child.label.name: child[0] for child in node if has_type(child, NodeType.ENT)}


def delete_id_column(
    session: neo4j.Session,
) -> None:
    """
    Delete the _architxt_group_id column from all nodes in the graph.

    :param session: Neo4j session.
    """
    session.run("MATCH (n) REMOVE n._architxt_group_id")

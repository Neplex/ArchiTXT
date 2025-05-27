import neo4j

from architxt.schema import Schema
from architxt.tree import Forest, NodeType, Tree, has_type


def export_graph(
    forest: Forest,
    session: neo4j.Session,
) -> None:
    """
    Export the graph instance as a dictionary using Neo4j.

    :param session: Neo4j session.
    :param forest: The forest to export.
    :return: A generator that yields dictionaries representing the graph.
    """
    schema = Schema.from_forest(forest)
    collapsible_nodes = schema.find_collapsible_groups()
    for tree in forest:
        export_tree(tree, session, collapsible_nodes)
    delete_id_column(session)


def export_tree(
    tree: Tree,
    session: neo4j.Session,
    edge_data: set[str],
) -> None:
    """
    Export the tree to the graph.

    :param tree: Tree to export.
    :param session: Neo4j session.
    :param edge_data:
    """
    for group in tree.subtrees(lambda st: has_type(st, NodeType.GROUP) and st.label.name not in edge_data):
        export_group(group, session)

    export_edge_data = {}
    for relation in tree.subtrees(lambda subtree: has_type(subtree, NodeType.REL)):
        if relation[0].label.name in edge_data:
            if relation[0] not in export_edge_data:
                export_edge_data[relation[0]] = set()
            export_edge_data[relation[0]].add(relation[1])

        elif relation[1].label.name in edge_data:
            if relation[1] not in export_edge_data:
                export_edge_data[relation[1]] = set()
            export_edge_data[relation[1]].add(relation[0])

        else:
            export_relation(relation, session)

    export_relation_edge_with_data(export_edge_data, session)


def export_relation(
    tree: Tree,
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

    rel_name = tree.metadata.get('source_column', tree.label.name.replace('<->', '_'))

    session.run(f"""
    MATCH (src:`{src.label.name}` {{_architxt_oid: '{src.oid}'}})
    MATCH (dest:`{dest.label.name}` {{_architxt_oid: '{dest.oid}'}})
    MERGE (src)-[r:`{rel_name}`]->(dest)
    """)


def export_relation_edge_with_data(
    edge_data: dict[Tree, set[Tree]],
    session: neo4j.Session,
) -> None:
    """
    Export the relation edge with data to the graph.

    :param edge_data: Dictionary of edges with data.
    :param session: Neo4j session.
    """
    for edge, relations in edge_data.items():
        src, dest = sorted(relations, key=lambda x: x.label)
        session.run(f"""
        MATCH (src:`{src.label.name}` {{_architxt_oid: '{src.oid}'}})
        MATCH (dest:`{dest.label.name}` {{_architxt_oid: '{dest.oid}'}})
        MERGE (src)-[r:{edge.label.name} {{ {', '.join(f'{k}: {v!r}' for k, v in get_properties(edge).items())} }}]->(dest)
        """)


def export_group(
    group: Tree,
    session: neo4j.Session,
) -> None:
    """
    Export the group to the graph.

    :param group: Group to export.
    :param session: Neo4j session.
    """
    session.run(f"CREATE INDEX _architxt_oid_index IF NOT EXISTS FOR (n:`{group.label.name}`) ON (n._architxt_oid)")
    session.run(
        f"""
        MERGE (n:`{group.label.name}` {{ _architxt_oid: $id }})
        ON CREATE SET n += $data
        """,
        id=str(group.oid),
        data=get_properties(group),
    )


def get_properties(node: Tree) -> dict[str, str]:
    """
    Get the properties of a node.

    :param node: Node to get properties from.
    :return: Dictionary of properties.
    """
    return {
        child.label.name: child[0] if isinstance(child[0], str | int | float | bool) else str(child[0])
        for child in node
        if has_type(child, NodeType.ENT)
    }


def delete_id_column(
    session: neo4j.Session,
) -> None:
    """
    Delete the _architxt_oid property from all nodes in the graph.

    :param session: Neo4j session.
    """
    session.run("MATCH (n) REMOVE n._architxt_oid")
    session.run("DROP INDEX _architxt_oid_index IF EXISTS")

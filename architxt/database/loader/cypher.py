from __future__ import annotations

from typing import TYPE_CHECKING

from neo4j.graph import Node, Relationship

from architxt.tree import NodeLabel, NodeType, Tree

if TYPE_CHECKING:
    from collections.abc import Generator

    from neo4j import Record, Session

__all__ = ['read_cypher']


def read_cypher(
    session: Session,
    *,
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Read the graph instance as a tree using Neo4j.

    :param session: Neo4j session.
    :param sample: Number of samples for each node to get.
    :return: A generator that yields trees representing the graph.
    """
    root_nodes = get_root_nodes(session, sample)

    relations_with_data = get_relation_with_data(session)
    for node in root_nodes:
        yield Tree(
            "ROOT", read_node(node, session=session, visited_relations=set(), relations_with_data=relations_with_data)
        )


def get_root_nodes(session: Session, sample: int) -> Generator[Node, None, None]:
    """
    Get the root nodes of the graph.

    :param session: Neo4j session.
    :param sample: Number of samples to get.
    :return: A generator of root nodes.
    """
    query = "MATCH (n) WHERE not ()-[]->(n) RETURN n"

    if sample > 0:
        query += " LIMIT $sample"

    for record in session.run(query, sample=sample):
        yield record['n']


def read_node(
    node: Node,
    *,
    session: Session,
    visited_relations: set[str],
    relations_with_data: set[str],
) -> Generator[Tree, None, None]:
    """
    Read the node and its children from the graph.

    :param node: Node to read.
    :param session: Neo4j session.
    :param visited_relations: Set of visited relations.
    :param relations_with_data: Set of relations with data.
    :return: A generator that yields trees representing the node.
    """
    yield build_group(node)

    for record in session.run(
        """
        MATCH (n)-[r]-(m)
        WHERE elementId(n) = $node_id AND NOT type(r) IN $visited_relations
        RETURN n, r, m
        """,
        node_id=node.element_id,
        visited_relations=list(visited_relations),
    ):
        visited_relations.add(record['r'].type)
        yield from parse_relation(
            record, session=session, visited_relations=visited_relations, relations_with_data=relations_with_data
        )


def get_relation_with_data(session: Session) -> set[str]:
    """
    Get the relations with data from the graph.

    :param session: Neo4j session.
    :return: A set of relations with data.
    """
    return {
        record['rtype']
        for record in session.run(
            """
            MATCH ()-[r]-()
            WITH type(r) AS rtype, collect(r) AS relations
            WHERE any(r IN relations WHERE size(keys(r)) > 0)
            RETURN rtype
            """
        )
    }


def parse_relation(
    record: Record,
    visited_relations: set[str],
    relations_with_data: set[str],
    *,
    session: Session,
) -> Generator[Tree, None, None]:
    """
    Parse a relation between two nodes.

    :param record: Record containing the relation.
    :param visited_relations: Set of visited relations.
    :param relations_with_data: Set of relations with data.
    :param session: Neo4j session.
    :return: A generator that yields trees representing the relation.
    """
    rel_name = record['r'].type

    if rel_name in relations_with_data:
        yield build_relation(record['r'], record['n'], rel_name)
        yield build_relation(record['r'], record['m'], rel_name)

    else:
        yield build_relation(record['n'], record['m'], rel_name)

    yield from read_node(
        record['m'], session=session, visited_relations=visited_relations, relations_with_data=relations_with_data
    )


def build_group(node: Node | Relationship) -> Tree:
    """
    Create a tree representation for a table with its columns and data.

    :param node: Node representing the table.
    :return: A tree representing the table's structure and data.
    """
    group_name = next(iter(node.labels)) if isinstance(node, Node) else node.type

    node_label = NodeLabel(NodeType.GROUP, group_name)
    entities = []
    for column in node:
        if not (entity_data := node[column]):
            continue

        entity_label = NodeLabel(NodeType.ENT, column)
        entity_tree = Tree(entity_label, [str(entity_data)])
        entities.append(entity_tree)

    return Tree(node_label, entities)


def build_relation(node1: Node, node2: Node, rel_name: str) -> Tree:
    """
    Create a tree representation for a relation.

    :param node1: The first node in the relation.
    :param node2: The second node in the relation.
    :param rel_name: The name of the relation.
    :return: A tree representing the relation between the two nodes.
    """
    return Tree(
        NodeLabel(NodeType.REL, rel_name),
        [
            build_group(node1),
            build_group(node2),
        ],
    )

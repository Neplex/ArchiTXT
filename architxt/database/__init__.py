import warnings
from collections.abc import Generator
from typing import Any

from sqlalchemy import Connection, ForeignKey, MetaData, Row, Table, exists
from tqdm.auto import tqdm

from architxt.tree import NodeLabel, NodeType, Tree

__all__ = ['read_database']


def read_database(
    conn: Connection,
    *,
    simplify_association: bool = True,
    search_all_instances: bool = False,
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Read the database instance as a tree.

    :param conn: SQLAlchemy connection to the database.
    :param simplify_association: Flag to simplify non attributed association tables.
    :param search_all_instances: Flag to search for all instances of database.
    :param sample: Number of samples for each table to get.
    :return: A list of trees representing the database.
    """
    metadata = MetaData()
    metadata.reflect(bind=conn)

    root_tables = get_root_tables(set(metadata.tables.values()))

    for table in root_tables:
        yield from read_table(table, conn=conn, simplify_association=simplify_association, sample=sample)

        if not search_all_instances:
            continue

        for foreign_table in table.foreign_keys:
            if foreign_table.column.table not in root_tables:
                yield from read_unreferenced_table(foreign_table, conn=conn, sample=sample)


def get_root_tables(tables: set[Table]) -> set[Table]:
    """
    Retrieve the root tables in the database by identifying tables that are not referenced as foreign keys.

    :param tables: A collection of tables to analyze.
    :return: A set of root table.
    """
    referenced_tables = {fk.column.table for table in tables for fk in table.foreign_keys}

    if not referenced_tables:
        return tables

    root_tables = tables - referenced_tables
    root_tables |= get_cycle_tables(referenced_tables)

    return root_tables


def get_cycle_tables(tables: set[Table]) -> set[Table]:
    """
    Retrieve tables that are part of a cycle in the database relations.

    If multiple tables are in a cycle, only the one with the maximum foreign keys is returned.

    :param tables: A collection of tables to analyze.
    :return: A set of tables that are part of a cycle but should be considered as root.
    """

    def get_cycle(table: Table, cycle: set[Table] | None = None) -> set[Table] | None:
        cycle = cycle or set()

        if table in cycle:
            return cycle

        for fk in table.foreign_keys:
            if cycle := get_cycle(fk.column.table, cycle | {table}):
                return cycle

        return None

    cycle_roots: set[Table] = set()
    referenced_tables = {fk.column.table for table in tables for fk in table.foreign_keys}

    while referenced_tables:
        table = referenced_tables.pop()

        if table_cycle := get_cycle(table):
            referenced_tables -= table_cycle
            selected_table = max(table_cycle, key=lambda x: len(x.foreign_keys))
            cycle_roots.add(selected_table)

    return cycle_roots


def is_association_table(table: Table) -> bool:
    """
    Check if a table is a many-to-many association table.

    :param table: The table to check.
    :return: True if the tale is a relation else False.
    """
    return len(table.foreign_keys) == len(table.primary_key.columns) == len(table.columns) == 2


def read_table(
    table: Table,
    *,
    conn: Connection,
    simplify_association: bool = False,
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Process the relations of a given table, retrieve data, and construct tree representations.

    :param table: The table to process.
    :param conn: SQLAlchemy connection.
    :param simplify_association: Flag to simplify non attributed association tables.
    :param sample: Number of samples for each table to get.
    :return: A list of trees representing the relations and data for the table.
    """
    association_table = simplify_association and is_association_table(table)
    query = table.select()

    if sample > 0:
        query = query.limit(sample)

    for row in tqdm(conn.execute(query), desc=table.name):
        if association_table:
            children = parse_association_table(table, row, conn=conn)
        else:
            children = parse_table(table, row, conn=conn)

        yield Tree("ROOT", children)


def read_unreferenced_table(
    foreign_key: ForeignKey,
    *,
    conn: Connection,
    sample: int = 0,
    _visited_links: set[ForeignKey] | None = None,
) -> Generator[Tree, None, None]:
    """
    Process the relations of a table that is not referenced by any other tables.

    :param foreign_key: The foreign key to process.
    :param conn: SQLAlchemy connection.
    :param sample: Number of samples for each table to get.
    :param _visited_links: Set of visited relations to avoid cycles.
    :return: A list of trees representing the relations and data for the table.
    """
    table = foreign_key.column.table

    query = table.select().where(~exists().where(foreign_key.parent == foreign_key.column))

    if sample > 0:
        query = query.limit(sample)

    for row in tqdm(conn.execute(query), desc=table.name):
        yield Tree("ROOT", parse_table(table, row, conn=conn))

    if _visited_links is None:
        _visited_links = set()

    _visited_links.add(foreign_key)
    for fk in table.foreign_keys:
        if fk.column.table != table:
            yield from read_unreferenced_table(fk, conn=conn, sample=sample, _visited_links=_visited_links)


def parse_association_table(
    table: Table,
    row: Row,
    *,
    conn: Connection,
) -> Generator[Tree, None, None]:
    """
    Parse a row of an association table into trees.

    The table is discarded and represented only as a relation between the two linked tables.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :yield: Trees representing the relations and data for the table.
    """
    left_fk, right_fk = table.foreign_keys
    left_row = conn.execute(
        left_fk.column.table.select().where(left_fk.column == row._mapping[left_fk.parent.name])
    ).fetchone()
    right_row = conn.execute(
        right_fk.column.table.select().where(right_fk.column == row._mapping[right_fk.parent.name])
    ).fetchone()

    if not left_row or not right_row:
        warnings.warn("Database have broken foreign keys!")
        return

    yield build_relation(
        left_table=left_fk.column.table,
        right_table=right_fk.column.table,
        left_row=left_row,
        right_row=right_row,
        name=table.name,
    )

    visited_links: set[ForeignKey] = set()
    yield from parse_table(left_fk.column.table, left_row, conn=conn, _visited_links=visited_links)
    yield from parse_table(right_fk.column.table, right_row, conn=conn, _visited_links=visited_links)


def parse_table(
    table: Table,
    row: Row,
    *,
    conn: Connection,
    _visited_links: set[ForeignKey] | None = None,
) -> Generator[Tree, None, None]:
    """
    Parse a row of a table into trees.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :param _visited_links: Set of visited relations to avoid cycles.
    :yield: Trees representing the relations and data for the table.
    """
    if _visited_links is None:
        _visited_links = set()

    yield build_group(table, row)

    for fk in table.foreign_keys:
        if fk in _visited_links:
            continue

        _visited_links.add(fk)

        yield from _parse_relation(table, row, fk, conn=conn, visited_links=_visited_links)


def _parse_relation(
    table: Table,
    row: Row,
    fk: ForeignKey,
    *,
    conn: Connection,
    visited_links: set[ForeignKey],
) -> Generator[Tree, None, None]:
    """
    Parse the relations for a table and construct a tree with the related data.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :param visited_links: Set of visited relations to avoid cycles.
    :return: A list of trees representing the relations and data for the table.
    """
    node_data = {"source": fk.parent.table.name, "target": fk.column.table.name, "source_column": fk.parent.name}
    linked_rows = fk.column.table.select().where(fk.column == row._mapping[fk.parent.name])

    for linked_row in conn.execute(linked_rows):
        yield build_relation(
            left_table=table,
            right_table=fk.column.table,
            left_row=row,
            right_row=linked_row,
            node_data=node_data,
        )

        yield from parse_table(
            fk.column.table,
            linked_row,
            conn=conn,
            _visited_links=visited_links,
        )


def build_group(table: Table, row: Row) -> Tree:
    """
    Create a tree representation for a table with its columns and data.

    :param table: The table to process.
    :param row: A row of the table.
    :return: A tree representing the table's structure and data.
    """
    primary_keys = {column.name for column in table.primary_key.columns}
    group_name = table.name.replace(' ', '')
    node_label = NodeLabel(NodeType.GROUP, group_name, {'primary_keys': primary_keys})

    entities = []
    for column in table.columns.values():
        if not (entity_data := row._mapping[column.name]):
            continue

        entity_name = column.name.replace(' ', '')
        entity_label = NodeLabel(
            NodeType.ENT,
            entity_name,
            {
                'type': column.type,
                'nullable': column.nullable,
                'default': column.default,
            },
        )
        entity_tree = Tree(entity_label, [str(entity_data)])
        entities.append(entity_tree)

    return Tree(node_label, entities)


def build_relation(
    left_table: Table,
    right_table: Table,
    left_row: Row,
    right_row: Row,
    node_data: dict[str, Any] | None = None,
    name: str = '',
) -> Tree:
    """
    Handle the current data for a table and its referred table.

    :param left_table: The left table of the relation.
    :param right_table: The right table of the relation.
    :param left_row: The left table row of the relation.
    :param right_row: The right table row of the relation.
    :param node_data: Dictionary containing relation data.
    :param name: Name of the relation, if not set, it will be automatically generated.
    :return: The tree of the relation.
    """
    if name:
        rel_name = name.replace(' ', '')

    else:
        left_name = left_table.name.replace(' ', '')
        right_name = right_table.name.replace(' ', '')
        rel_name = f"{left_name}<->{right_name}"

    return Tree(
        NodeLabel(NodeType.REL, rel_name, node_data),
        [
            build_group(left_table, left_row),
            build_group(right_table, right_row),
        ],
    )

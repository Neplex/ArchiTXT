from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING, Any

from sqlalchemy import Connection, ForeignKey, MetaData, Row, Table, exists, func, select
from tqdm.auto import tqdm

from architxt.tree import NodeLabel, NodeType, Tree

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ['read_sql']


def read_sql(
    conn: Connection,
    *,
    simplify_association: bool = True,
    search_all_instances: bool = False,
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Read the database instance as a tree.

    :param conn: SQLAlchemy connection to the database.
    :param simplify_association: Flag to simplify non-attributed association tables.
    :param search_all_instances: Flag to search for all instances of the database.
    :param sample: Number of samples for each table to get.
    :return: A list of trees representing the database.
    """
    metadata = MetaData()
    metadata.reflect(bind=conn)

    ns = uuid.uuid5(uuid.NAMESPACE_URL, conn.engine.url.render_as_string())
    root_tables = get_root_tables(set(metadata.tables.values()))

    for table in root_tables:
        yield from read_table(table, conn=conn, simplify_association=simplify_association, namespace=ns, sample=sample)

        if not search_all_instances:
            continue

        for foreign_table in table.foreign_keys:
            if foreign_table.column.table not in root_tables:
                yield from read_unreferenced_table(foreign_table, conn=conn, namespace=ns, sample=sample)


def get_root_tables(tables: set[Table]) -> set[Table]:
    """
    Retrieve the root tables in the database by identifying tables that are not referenced as foreign keys.

    :param tables: A collection of tables to analyze.
    :return: A set of root tables.
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

    def get_cycle(table: Table, _cycle: set[Table] | None = None) -> set[Table]:
        cycle = _cycle or set()

        if table in cycle:
            return cycle

        for fk in table.foreign_keys:
            if cycle := get_cycle(fk.column.table, cycle | {table}):
                return cycle

        return set()

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
    namespace: uuid.UUID,
    simplify_association: bool = False,
    sample: int = 0,
) -> Generator[Tree, None, None]:
    """
    Process the relations of a given table, retrieve data, and construct tree representations.

    :param table: The table to process.
    :param conn: SQLAlchemy connection.
    :param namespace: The database namespace to use for the object identifier.
    :param simplify_association: Flag to simplify non-attributed association tables.
    :param sample: Number of samples for each table to get.
    :return: A list of trees representing the relations and data for the table.
    """
    total_rows = conn.scalar(select(func.count()).select_from(table)) or 0
    query = table.select()

    if total_rows > sample > 0:
        query = query.limit(sample)
        total_rows = sample

    for row in tqdm(conn.execute(query), desc=table.name, total=total_rows):
        if simplify_association and is_association_table(table):
            children = parse_association_table(table, row, conn=conn, namespace=namespace)
        else:
            children = parse_table(table, row, conn=conn, namespace=namespace)

        yield Tree("ROOT", children)


def read_unreferenced_table(
    foreign_key: ForeignKey,
    *,
    conn: Connection,
    namespace: uuid.UUID,
    sample: int = 0,
    _visited_links: set[ForeignKey] | None = None,
) -> Generator[Tree, None, None]:
    """
    Process the relations of a table that is not referenced by any other tables.

    :param foreign_key: The foreign key to process.
    :param conn: SQLAlchemy connection.
    :param namespace: The database namespace to use for the object identifier.
    :param sample: Number of samples for each table to get.
    :param _visited_links: Set of visited relations to avoid cycles.
    :return: A list of trees representing the relations and data for the table.
    """
    table = foreign_key.column.table

    query = table.select().where(~exists().where(foreign_key.parent == foreign_key.column))

    if sample > 0:
        query = query.limit(sample)

    for row in tqdm(conn.execute(query), desc=table.name):
        yield Tree("ROOT", parse_table(table, row, conn=conn, namespace=namespace))

    if _visited_links is None:
        _visited_links = set()

    _visited_links.add(foreign_key)
    for fk in table.foreign_keys:
        if fk.column.table != table:
            yield from read_unreferenced_table(
                fk, conn=conn, sample=sample, namespace=namespace, _visited_links=_visited_links
            )


def parse_association_table(
    table: Table,
    row: Row,
    *,
    conn: Connection,
    namespace: uuid.UUID,
) -> Generator[Tree, None, None]:
    """
    Parse a row of an association table into trees.

    The table is discarded and represented only as a relation between the two linked tables.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :param namespace: The database namespace to use for the object identifier.
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
        namespace=namespace,
    )

    visited_links: set[ForeignKey] = set()
    yield from parse_table(left_fk.column.table, left_row, conn=conn, namespace=namespace, _visited_links=visited_links)
    yield from parse_table(
        right_fk.column.table, right_row, conn=conn, namespace=namespace, _visited_links=visited_links
    )


def parse_table(
    table: Table,
    row: Row,
    *,
    conn: Connection,
    namespace: uuid.UUID,
    _visited_links: set[ForeignKey] | None = None,
) -> Generator[Tree, None, None]:
    """
    Parse a row of a table into trees.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :param namespace: The database namespace to use for the object identifier.
    :param _visited_links: Set of visited relations to avoid cycles.
    :yield: Trees representing the relations and data for the table.
    """
    if _visited_links is None:
        _visited_links = set()

    yield build_group(table, row, namespace=namespace)

    for fk in sorted(table.foreign_keys, key=lambda x: x.parent.name):
        if fk in _visited_links:
            continue

        _visited_links.add(fk)

        yield from _parse_relation(table, row, fk, conn=conn, namespace=namespace, visited_links=_visited_links)


def _parse_relation(
    table: Table,
    row: Row,
    fk: ForeignKey,
    *,
    conn: Connection,
    namespace: uuid.UUID,
    visited_links: set[ForeignKey],
) -> Generator[Tree, None, None]:
    """
    Parse the relations for a table and construct a tree with the related data.

    :param table: The table to process.
    :param row: A row of the table.
    :param conn: SQLAlchemy connection.
    :param namespace: The database namespace to use for the object identifier.
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
            namespace=namespace,
        )

        yield from parse_table(
            fk.column.table,
            linked_row,
            conn=conn,
            namespace=namespace,
            _visited_links=visited_links,
        )


def build_group(table: Table, row: Row, namespace: uuid.UUID) -> Tree:
    """
    Create a tree representation for a table with its columns and data.

    :param table: The table to process.
    :param row: A row of the table.
    :param namespace: The database namespace to use for the object identifier.
    :return: A tree representing the table's structure and data.
    """
    primary_keys = {column.name for column in table.primary_key.columns}
    group_name = table.name.replace(' ', '_')
    node_label = NodeLabel(NodeType.GROUP, group_name)
    primary_data: dict[str, Any] = {}

    entities = []
    for column in table.columns.values():
        entity_data = row._mapping[column.name]

        if column.name in primary_keys:
            primary_data[column.name] = entity_data

        if entity_data is None or column.foreign_keys:
            continue

        entity_name = column.name.replace(' ', '_')
        entity_label = NodeLabel(NodeType.ENT, entity_name)
        entity_tree = Tree(
            entity_label,
            [str(entity_data)],
            {
                'type': column.type,
                'nullable': column.nullable,
                'default': column.default,
            },
        )
        entities.append(entity_tree)

    return Tree(
        node_label,
        entities,
        {
            'primary_keys': primary_keys,
        },
        oid=get_oid(namespace, group_name, primary_data),
    )


def build_relation(
    left_table: Table,
    right_table: Table,
    left_row: Row,
    right_row: Row,
    namespace: uuid.UUID,
    node_data: dict[str, Any] | None = None,
    name: str = '',
) -> Tree:
    """
    Handle the current data for a table and its referred table.

    :param left_table: The left table of the relation.
    :param right_table: The right table of the relation.
    :param left_row: The left table row of the relation.
    :param right_row: The right table row of the relation.
    :param namespace: The database namespace to use for the object identifier.
    :param node_data: Dictionary containing relation data.
    :param name: The name of the relation, if not set, it will be automatically generated.
    :return: The tree of the relation.
    """
    if name:
        rel_name = name.replace(' ', '_')

    else:
        left_name = left_table.name.replace(' ', '_')
        right_name = right_table.name.replace(' ', '_')
        rel_name = f'{left_name}<->{right_name}'

    # Get primary key values from both tables
    primary_data = {f'left_{col.name}': left_row._mapping[col.name] for col in left_table.primary_key.columns} | {
        f'right_{col.name}': right_row._mapping[col.name] for col in right_table.primary_key.columns
    }

    return Tree(
        NodeLabel(NodeType.REL, rel_name),
        [
            build_group(left_table, left_row, namespace=namespace),
            build_group(right_table, right_row, namespace=namespace),
        ],
        node_data,
        oid=get_oid(namespace, rel_name, primary_data),
    )


def get_oid(namespace: uuid.UUID, name: str, data: dict[str, Any]) -> uuid.UUID:
    """
    Generate an object identifier based on the DB namespace, the name of the table/relation, and primary key values.

    The namespace hierarchy follows this structure::

        Database Namespace
        └── Table/Relation Namespace (uuid5(db_namespace, name))
            └── Record OID (uuid5(table_namespace, sorted_data))

    :param namespace: UUID namespace to use as base for generation
    :param name: Base name for the identifier (table or relation name)
    :param data: Dictionary of primary key values used to generate unique identifier
    :return: UUID5 generated from the namespace and combined name/data
    """
    namespace = uuid.uuid5(namespace, name)
    data_str = ';'.join(f'{key}={data[key]}' for key in sorted(data))

    return uuid.uuid5(namespace, data_str)

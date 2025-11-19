from __future__ import annotations

import base64
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BLOB,
    Column,
    Connection,
    Date,
    DateTime,
    ForeignKey,
    MetaData,
    String,
    Table,
    Uuid,
    insert,
)
from tqdm.auto import tqdm

from architxt.schema import Group, Relation, RelationOrientation, Schema
from architxt.tree import Forest, NodeType, Tree, TreeOID, has_type

if TYPE_CHECKING:
    from architxt.tree import _TypedSubTree, _TypedTree

__all__ = ['export_sql']

PKColumnFactory = Callable[[str], str]


def default_pk_factory(
    table_name: str,
) -> str:
    """
    Generate the ID column for the given table.

    :param table_name: The table name to generate ID for.
    :return: The name of the ID column for the table.
    """
    return f'architxt_{table_name}ID'


def export_sql(
    forest: Forest,
    conn: Connection,
    *,
    pk_factory: PKColumnFactory = default_pk_factory,
) -> None:
    """
    Export the forest to the relational database.

    :param conn: Connection to the relational database.
    :param forest: Forest to export.
    :param pk_factory: A column name factory for the groups primary keys.
    """
    schema = Schema.from_forest(forest, keep_unlabelled=False)
    create_schema(conn, schema, pk_factory)

    for tree in tqdm(forest, desc="Exporting relational database"):
        export_tree(tree, conn, schema, pk_factory)
        conn.commit()


def create_schema(
    conn: Connection,
    schema: Schema,
    pk_factory: PKColumnFactory,
) -> Schema:
    """
    Create the schema for the relational database.

    :param conn: Connection to the graph.
    :param schema: The schema to build.
    :param pk_factory: A column name factory for the groups primary keys.
    """
    metadata = MetaData()

    database_schema: dict[str, Table] = {}
    for group in schema.groups:
        database_schema[group.name] = create_table_for_group(group, metadata, pk_factory)

    for rel in schema.relations:
        if rel.orientation == RelationOrientation.BOTH:
            create_table_for_relation(database_schema, rel, metadata, pk_factory)

        else:
            add_foreign_keys_to_table(database_schema, rel, pk_factory)

    metadata.create_all(conn)
    return schema


def create_table_for_group(group: Group, metadata: MetaData, pk_factory: PKColumnFactory) -> Table:
    """
    Create a table for the given group.

    :param group: The group to create a table for.
    :param metadata: SQLAlchemy metadata to attach the table to.
    :param pk_factory: A column name factory for the groups primary keys.
    :return: SQLAlchemy Table object.
    """
    return Table(
        group.name,
        metadata,
        Column(pk_factory(group.name), Uuid, primary_key=True),
        *(Column(entity, String) for entity in group.entities),
    )


def add_foreign_keys_to_table(
    database_schema: dict[str, Table],
    relation: Relation,
    pk_factory: PKColumnFactory,
) -> None:
    """
    Add foreign key constraints to the database schema.

    :param database_schema: The dictionary of tables in the database schema.
    :param relation: The relation to build as a foreign key.
    :param pk_factory: A column name factory for the groups primary keys.
    """
    left = database_schema[relation.left.replace(' ', '')]
    right = database_schema[relation.right.replace(' ', '')]

    source, target = (left, right) if relation.orientation == RelationOrientation.LEFT else (right, left)

    column_name = relation.name if source.name == target.name else pk_factory(target.name)
    target_column_name = target.primary_key.columns.keys()[0]

    database_schema[source.name].append_column(Column(column_name, ForeignKey(f"{target.name}.{target_column_name}")))


def create_table_for_relation(
    database_schema: dict[str, Table],
    relation: Relation,
    metadata: MetaData,
    pk_factory: PKColumnFactory,
) -> None:
    """
    Create a table for the given relation.

    :param database_schema: The dictionary of tables in the database schema.
    :param relation: The relation to build the table for.
    :param metadata: SQLAlchemy metadata to attach the table to.
    :param pk_factory: A column name factory for the groups primary keys.
    """
    left = database_schema[relation.left.replace(" ", "")]
    right = database_schema[relation.right.replace(" ", "")]
    left_key = pk_factory(left.name)
    right_key = pk_factory(right.name)

    database_schema[relation.name] = Table(relation.name, metadata)
    database_schema[relation.name].append_column(
        Column(left_key, ForeignKey(f"{left.name}.{left_key}"), primary_key=True)
    )
    database_schema[relation.name].append_column(
        Column(right_key, ForeignKey(f"{right.name}.{right_key}"), primary_key=True)
    )


def export_tree(
    tree: Tree,
    conn: Connection,
    schema: Schema,
    pk_factory: PKColumnFactory,
) -> None:
    """
    Export the tree to the relational database.

    :param tree: Tree to export.
    :param conn: Connection to the relational database.
    :param schema: The schema.
    :param pk_factory: A column name factory for the groups primary key.
    """
    data_to_export: dict[str, dict[TreeOID, dict[str, str]]] = {}

    for subtree in tree.subtrees():
        if has_type(subtree, NodeType.GROUP):
            export_group(subtree, data_to_export, pk_factory)

        elif has_type(subtree, NodeType.REL):
            export_relation(subtree, data_to_export, schema, pk_factory)

    export_data(data_to_export, conn)


def export_relation(
    tree: _TypedTree,
    data: dict[str, dict[TreeOID, dict[str, str]]],
    schema: Schema,
    pk_factory: PKColumnFactory,
) -> None:
    """
    Export the relation to the relational database.

    :param tree: Relation to export.
    :param data: Data to export.
    :param schema: The schema.
    :param pk_factory: A column name factory for the groups primary key.
    """
    relation = next(rel for rel in schema.relations if rel.name == tree.label.name)

    if relation.orientation == RelationOrientation.BOTH:
        relation_data: dict[str, str] = {}
        for child in tree:
            column_name = pk_factory(child.label.name)
            relation_data[column_name] = data[child.label.name][child.oid][column_name]

        data[relation.name] = {tree.oid: relation_data}

    else:
        left: _TypedSubTree | None = None
        right: _TypedSubTree | None = None

        for child in tree:
            if not has_type(child, NodeType.GROUP):
                continue

            if child.label.name == relation.left:
                left = child

            elif child.label.name == relation.right:
                right = child

        if not left or not right:
            return

        if relation.orientation == RelationOrientation.RIGHT:
            left, right = right, left

        column_name = relation.name if right.label.name == left.label.name else pk_factory(right.label.name)
        data[left.label.name][left.oid][column_name] = data[right.label.name][right.oid][column_name]


def export_group(
    group: _TypedTree,
    data: dict[str, dict[TreeOID, dict[str, str]]],
    pk_factory: PKColumnFactory,
) -> None:
    """
    Export the group to the relational database.

    :param group: Group to export.
    :param data: Data to export.
    :param pk_factory: A column name factory for the groups primary key.
    """
    group_name = group.label.name

    group_data = get_data_from_group(group)
    group_data[pk_factory(group_name)] = str(group.oid)

    if group_name not in data:
        data[group_name] = {}

    data[group_name][group.oid] = group_data


def get_data_from_group(group: Tree) -> dict[str, str]:
    """
    Get data from the relational database.

    :param group: Group to get data from.
    :return: Data from the group.
    """
    data: dict[str, str] = {}

    for entity in group.subtrees():
        if not has_type(entity, NodeType.ENT):
            continue

        value = entity.metadata.get('value') or ' '.join(str(leaf) for leaf in entity.leaves())
        dt = entity.metadata.get('type')

        if isinstance(dt, Date):
            value = datetime.strptime(value, '%Y-%m-%d').date()

        elif isinstance(dt, DateTime):
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')

        elif isinstance(dt, BLOB):
            value = base64.b64decode(value)

        elif value.lower() in {'true', 'false'}:
            value = value.lower() == 'true'

        else:
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except (ValueError, TypeError):
                    continue

        data[entity.label.name] = value

    return data


def export_data(
    data: dict[str, dict[TreeOID, dict[str, str]]],
    conn: Connection,
) -> None:
    """
    Export the data to the relational database.

    :param data: Data to export.
    :param conn: Connection to the relational database.
    """
    if not data:
        return

    data_to_export: dict[str, dict[TreeOID, dict[str, str]]] = defaultdict(dict)
    table_to_insert: dict[str, list[dict[str, str]]] = defaultdict(list)

    for table, dict_info in data.items():
        for oid, info in dict_info.items():
            has_foreign_key = False

            for name, x in info.items():
                if isinstance(x, dict) and 'primary_key_insert' not in x:
                    has_foreign_key = True
                elif isinstance(x, dict):
                    data[table][oid][name] = x['primary_key_insert']

            if has_foreign_key:
                data_to_export[table][oid] = info
            else:
                table_to_insert[table].append(info)

    export_table_to_insert(table_to_insert, conn)

    export_data(data_to_export, conn)


def export_table_to_insert(
    table_to_insert: dict[str, list[dict[str, str]]],
    conn: Connection,
) -> None:
    """
    Export the table to the graph.

    :param table_to_insert: Tables to insert.
    :param conn: Connection to the graph.
    """
    for table in table_to_insert:
        for row in table_to_insert[table]:
            info = row
            database_table = Table(table, MetaData(), autoload_with=conn)

            primary_keys = [col.name for col in database_table.primary_key.columns]
            query = (
                database_table.select()
                .with_only_columns(*[getattr(database_table.c, key) for key in primary_keys])
                .where(*[getattr(database_table.c, key) == value for key, value in info.items() if key in primary_keys])
            )
            result = conn.execute(query).fetchone()

            if not result:
                insert_command = insert(database_table).values(info)
                result_insert = conn.execute(insert_command)

                inserted_id = result_insert.inserted_primary_key[0]
            else:
                inserted_id = result[0]

            if inserted_id:
                info['primary_key_insert'] = inserted_id

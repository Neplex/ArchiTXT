import base64
from datetime import datetime

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
    insert,
)
from tqdm.auto import tqdm

from architxt.schema import Schema
from architxt.tree import NodeLabel, NodeType, Tree, has_type


def export_relational(
    conn: Connection,
    *,
    forest: list[Tree],
) -> None:
    """
    Export the forest to the relational database.

    :param conn: Connection to the relational database.
    :param forest: Forest to export.
    """
    table_relation = create_schema(conn=conn, forest=forest)
    for tree in tqdm(forest, desc="Exporting relational database"):
        export_tree(tree, conn=conn, table_relation=table_relation)
        conn.commit()


def create_schema(
    conn: Connection,
    *,
    forest: list[Tree],
) -> dict[str, dict[str, str]]:
    """
    Create the schema for the relational database.

    :param conn: Connection to the graph.
    :param forest:
    """
    schema = Schema.from_forest(forest, keep_unlabelled=False)
    metadata = MetaData()
    database_schema = {}
    for group, children in schema.groups.items():
        database_schema[group.name] = create_table_for_group(group, metadata, children)

    relation = get_relation_from_forest(forest, schema)
    for group, children in schema.relations.items():
        if group.name in relation["1-n"]:
            add_foreign_keys_to_table(database_schema, group, children, relation)
        else:
            create_table_for_relation(database_schema, children, metadata)

    metadata.create_all(conn)
    return relation


def create_table_for_group(group: NodeLabel, metadata: MetaData, children: set[NodeLabel]) -> Table:
    """
    Create a table for the given group.

    :param group: The group (NodeLabel) to create a table for.
    :param metadata: SQLAlchemy metadata to attach the table to.
    :param children: The child nodes of the group.
    :return: SQLAlchemy Table object.
    """
    columns = create_all_columns(group, children)
    return Table(group.name, metadata, *columns)


def add_foreign_keys_to_table(
    database_schema: dict, group: NodeLabel, children: tuple[NodeLabel, NodeLabel], relation: dict
) -> None:
    """
    Add foreign key constraints to the database schema.

    :param database_schema: The dictionary of tables in the database schema.
    :param group: The group (NodeLabel) defining the relation.
    :param children: The child nodes related to the group.
    """
    if relation["1-n"][group.name]["source"]:
        source = database_schema[children[1].name.replace(" ", "")]
        target = database_schema[children[0].name.replace(" ", "")]
    else:
        source = database_schema[children[0].name.replace(" ", "")]
        target = database_schema[children[1].name.replace(" ", "")]

    column_name = group.name if source.name == target.name else target.name + "ID"
    target_column_name = target.primary_key.columns.keys()[0]

    database_schema[source.name].append_column(Column(column_name, ForeignKey(f"{target.name}.{target_column_name}")))


def create_table_for_relation(
    database_schema: dict,
    children: tuple[NodeLabel, NodeLabel],
    metadata: MetaData,
) -> None:
    """
    Create a table for the given relation.

    :param database_schema: The dictionary of tables in the database schema.
    :param group: The group (NodeLabel) defining the relation.
    :param children: The child nodes related to the group.
    """
    group1 = database_schema[children[0].name.replace(" ", "")]
    group2 = database_schema[children[1].name.replace(" ", "")]

    group_name = group1.name + "_" + group2.name
    database_schema[group_name] = Table(group_name, metadata)
    database_schema[group_name].append_column(
        Column(group1.name + "ID", ForeignKey(f"{group1.name}.Architxt_{group1.name + 'ID'}"), primary_key=True)
    )
    database_schema[group_name].append_column(
        Column(group2.name + "ID", ForeignKey(f"{group2.name}.Architxt_{group2.name + 'ID'}"), primary_key=True)
    )


def get_relation_from_forest(
    forest: list[Tree],
    schema: Schema,
) -> dict[str, dict[str, str]]:
    """
    Get the relation from the forest.

    :param forest: Forest to get the relation from.
    :param schema: Schema to get the relation from.
    :return: Relation from the forest.
    """
    relations = schema.get_relations_type(forest=forest)
    return {"n-n": relations[0], "1-n": relations[1]}


def create_all_columns(
    group: NodeLabel,
    children: set[NodeLabel],
) -> list[Column]:
    """
    Create all columns for a group.

    :param group: Group to create columns for.
    :param children: Children of the group.
    :return: List of columns.
    """
    columns = [Column(child.name, String) for child in children]
    columns.append(Column(f'Architxt_{group.name}ID', String(36), primary_key=True))
    return columns


def export_tree(
    tree: Tree,
    *,
    conn: Connection,
    table_relation: dict[str, dict[str, str]],
) -> None:
    """
    Export the tree to the relational database.

    :param tree: Tree to export.
    :param conn: Connection to the relational database.
    :param table_relation: The relation between the tables.
    """
    data_to_export = {}

    for group in tree.subtrees(lambda subtree: has_type(subtree, NodeType.GROUP)):
        export_group(group, data=data_to_export)

    for relation in tree.subtrees(lambda subtree: has_type(subtree, NodeType.REL)):
        export_relation(relation, data=data_to_export, table_relation=table_relation)

    export_data(data=data_to_export, conn=conn)


def export_relation(
    tree: Tree,
    *,
    data: dict[str, dict[str, any]],
    table_relation: dict[str, dict[str, str]],
) -> None:
    """
    Export the relation to the relational database.

    :param tree: Relation to export.
    :param data: Data to export.
    :param table_relation: The relation between the tables.
    """
    rel_name = tree.label.name
    if rel_name in table_relation["n-n"]:
        table_name = tree[0].label.name + "_" + tree[1].label.name
        data[table_name] = {str(tree[0].oid): {}}
        data[table_name][str(tree[0].oid)][tree[0].label.name + "ID"] = data[tree[0].label.name][str(tree[0].oid)]
        data[table_name][str(tree[0].oid)][tree[1].label.name + "ID"] = data[tree[1].label.name][str(tree[1].oid)]
        return
    if rel_name in table_relation["1-n"]:
        if table_relation["1-n"][rel_name]["source"]:
            source = tree[1]
            target = tree[0]
        else:
            source = tree[0]
            target = tree[1]

    column_name = rel_name if target.label.name == source.label.name else target.label.name + "ID"

    data[source.label.name][str(source.oid)][column_name] = data[target.label.name][str(target.oid)]


def export_group(
    group: Tree,
    *,
    data: dict[str, dict[str, str]],
) -> None:
    """
    Export the group to the relational database.

    :param group: Group to export.
    :param data: Data to export.
    """
    group_name = group.label.name

    insert = get_data_from_group(group)
    insert["Architxt_" + group_name + "ID"] = str(group.oid)

    if group_name not in data:
        data[group_name] = {}
    data[group_name][str(group.oid)] = insert


def get_data_from_group(group: Tree) -> dict[str, str]:
    """
    Get data from the relational database.

    :param group: Group to get data from.
    :return: Data from the group.
    """
    result = {}
    for column in group:
        if column.label.name is None:
            continue
        if column.metadata and isinstance(column.metadata["type"], Date) and isinstance(column[0], str):
            column[0] = datetime.strptime(column[0], "%Y-%m-%d").date()
        elif column.metadata and isinstance(column.metadata["type"], DateTime) and isinstance(column[0], str):
            column[0] = datetime.strptime(column[0], "%Y-%m-%d %H:%M:%S")
        elif column.metadata and isinstance(column.metadata["type"], BLOB) and isinstance(column[0], str):
            column[0] = base64.b64decode(column[0])
        result[column.label.name] = column[0]
    return result


def export_data(
    data: dict,
    *,
    conn: Connection,
) -> None:
    """
    Export the data to the relational database.

    :param data: Data to export.
    :param conn: Connection to the relational database.
    :return:
    """
    if not data:
        return
    data_to_export = {}
    table_to_insert = {}
    for table, dict_info in data.items():
        for oid, info in dict_info.items():
            has_foreign_key = False
            for name, x in info.items():
                if isinstance(x, dict) and "primary_key_insert" not in x:
                    has_foreign_key = True
                elif isinstance(x, dict):
                    data[table][oid][name] = x["primary_key_insert"]
            if not has_foreign_key:
                if table not in table_to_insert:
                    table_to_insert[table] = []
                table_to_insert[table].append(info)
            else:
                if table not in data_to_export:
                    data_to_export[table] = {}
                data_to_export[table][oid] = info

    export_table_to_insert(table_to_insert=table_to_insert, conn=conn)

    export_data(data=data_to_export, conn=conn)


def export_table_to_insert(
    table_to_insert: dict[str, list[dict[str, str]]],
    *,
    conn: Connection,
) -> None:
    """
    Export the table to the graph.

    :param table_to_insert: Tables to insert.
    :param data: Data to insert.
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
                info["primary_key_insert"] = inserted_id

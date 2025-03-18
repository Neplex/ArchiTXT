from collections.abc import Generator
from typing import Any

from sqlalchemy import MetaData, create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from architxt.tree import Forest, NodeLabel, NodeType, Tree


def read_database(
    db_connection: str,
    remove_many_to_many: bool = True,
) -> Forest:
    """
    Read the database, retrieve table information, process table relations, and print the results in a tree format.

    :param db_connection: Connection string for the database.
    :param remove_many_to_many: Flag to remove many-to-many tables.
    :return: A list of trees representing the database.
    """
    # Create a connection to the SQLite database
    engine = create_engine(db_connection)
    inspector = inspect(engine)
    session_maker = sessionmaker(bind=engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    with session_maker() as session:
        tables = inspector.get_table_names()

        table_info = read_info_database(metadata)
        primary_keys = table_info.get("Primary keys", {})
        table_relations = table_info.get("Foreign keys", {})
        parent_tables = get_parent_tables(table_relations, tables)

        if remove_many_to_many:
            parent_tables, table_relations = remove_many_to_many_tables(parent_tables, table_relations, inspector)

        final_forest = []
        for table in parent_tables:
            final_forest.extend(
                process_table_relations(table, table_relations, session, inspector, metadata, primary_keys)
            )

        for tree in final_forest:
            print(tree.pformat(margin=255), end="\n\n\n")
            print(tree)


def read_info_database(
    meta: MetaData,
) -> dict[str, dict[str, str]]:
    """
    Retrieve information about the database tables, including primary and foreign keys.

    :param meta: SQLAlchemy MetaData object.
    :return: A dictionary containing primary keys and foreign keys for each table.
    """
    table_info = {"Primary keys": {}, "Foreign keys": {}}

    for table_name, table in meta.tables.items():
        table_info["Primary keys"][table_name] = [pk.name for pk in table.primary_key.columns]

        foreign_keys = [
            {
                "column": fk.parent.name,
                "referred_table": fk.column.table.name,
                "referred_column": fk.column.name,
            }
            for fk in table.foreign_keys
        ]
        if foreign_keys:
            table_info["Foreign keys"][table_name] = foreign_keys
    return table_info


def get_parent_tables(relations: dict[str, list[dict[str, str]]], tables: list[str]) -> set[str]:
    """
    Retrieve the parent tables in the database by identifying tables that are not referenced as foreign keys.

    :param relations: A dictionary of foreign key relations.
    :param tables: A list of table names.
    :return: A set of parent table names.
    """
    childrens = {value["referred_table"] for info in relations.values() for value in info}

    if not childrens:
        return set(tables)

    parents = set(relations.keys()) - childrens
    cycle_tables = get_cycle_tables(relations)
    if cycle_tables:
        parents.add(cycle_tables.pop())

    return parents


def get_cycle_tables(
    relations: dict[str, list[dict[str, str]]],
) -> set[str]:
    """
    Retrieve tables that are part of a cycle in the database relations.

    :param relations: A dictionary of foreign key relations.
    :return: A set of table names that are part of a cycle.
    """
    cycle_tables = set()
    for table, info in relations.items():
        for value in info:
            if value['referred_table'] == table:
                continue
            if value['referred_table'] in relations and table in [
                x['referred_table'] for x in relations[value['referred_table']]
            ]:
                cycle_tables.add(table)
                cycle_tables.add(value['referred_table'])
    return cycle_tables


def remove_many_to_many_tables(
    parent_tables: set[str],
    relations: dict[str, list[dict[str, str]]],
    inspector: inspect,
) -> tuple[set[str], dict[str, list[dict[str, str]]]]:
    """
    Remove many-to-many tables from the relations.

    :param parent_tables: Set of parent table names.
    :param relations: A dictionary of foreign key relations.
    :param inspector: SQLAlchemy inspector object.
    :return: Updated parent_tables and relations.
    """
    for table in identify_many_to_many_tables(parent_tables, relations, inspector):
        parent_tables.remove(table)
        for relation in relations[table]:
            for relation2 in relations[table]:
                if relation["referred_table"] == relation2["referred_table"]:
                    continue
                if relation["referred_table"] not in relations:
                    relations[relation["referred_table"]] = []
                new_relation = {
                    "column": relation["column"],
                    "referred_table": relation2["referred_table"],
                    "referred_column": relation["column"],
                    "table_to_many": table,
                }
                relations[relation["referred_table"]].append(new_relation)
        relations.pop(table)
    return parent_tables, relations


def identify_many_to_many_tables(
    parent_tables: set[str], relations: dict[str, list[dict[str, str]]], inspector: inspect
) -> set[str]:
    """
    Identify many-to-many tables from the parent tables.

    :param parent_tables: Set of parent table names.
    :param relations: A dictionary of foreign key relations.
    :param inspector: SQLAlchemy inspector object.
    :return: A set of many-to-many table names.
    """
    table_to_remove = set()
    for table in parent_tables:
        if len(relations.get(table, [])) == len(inspector.get_columns(table)):
            table_to_remove.add(table)
    return table_to_remove


def process_table_relations(
    table: str,
    relations: dict[str, list[dict[str, str]]],
    session: sessionmaker,
    inspector: inspect,
    metadata: MetaData,
    primary_keys: dict[str, list[str]],
) -> Generator[Tree, Any, None]:
    """
    Process the relations of a given table, retrieve data, and construct tree representations.

    :param table: Name of the table to process.
    :param relations: Dictionary of foreign key relations.
    :param session: SQLAlchemy session object.
    :param inspector: SQLAlchemy inspector object.
    :param metadata: SQLAlchemy MetaData object.
    :param primary_keys: Dictionary of primary keys for tables.
    :return: A list of trees representing the relations and data for the table.
    """
    meta_table = metadata.tables.get(table)
    info = session.execute(select(meta_table).limit(10)).fetchall()

    for data in info:
        root = Tree("ROOT", [])
        for tree in write_parse_relations(table, relations, data, primary_keys, session, inspector, metadata, set()):
            root.append(tree)
        yield root


def write_group_table(
    table: str,
    columns: list[dict[str, Any]],
    data: list[tuple],
    primary_keys: dict[str, list[str]] | None,
) -> Tree:
    """
    Create a tree representation for a table with its columns and data.

    :param table: Name of the table.
    :param columns: List of column dictionaries.
    :param data: Data for each row in the table.
    :param primary_keys: Dictionary of primary keys for tables.
    :return: A tree representing the table's structure and data.
    """
    node = NodeLabel(NodeType.GROUP, table, {"primary_keys": primary_keys})
    root = Tree(node, [])

    for i, column in enumerate(columns):
        node_data = {"type": column['type'], "nullable": column['nullable'], "default": column['default']}
        root.append(Tree(NodeLabel(NodeType.ENT, column['name'], node_data), [data[i]]))
    return root


def write_parse_relations(
    table: str,
    relations: dict[str, list[dict[str, str]]],
    data: list[tuple],
    primary_keys: dict[str, list[str]],
    session: sessionmaker,
    inspector: inspect,
    metadata: MetaData,
    visited_relations: set[str],
) -> Generator[Tree, Any, None]:
    """
    Parse the relations for a table and construct a tree with the related data.

    :param table: Name of the table.
    :param relations: Dictionary of foreign key relations.
    :param data: Data for the current row.
    :param primary_keys: Dictionary of primary keys for tables.
    :param session: SQLAlchemy session object.
    :param inspector: SQLAlchemy inspector object.
    :param metadata: SQLAlchemy MetaData object.
    :param visited_relations: Set of visited relations to avoid cycles.
    :return: A list of trees representing the relations and data for the table.
    """
    if table not in relations:
        return
    if table in visited_relations:
        return
    visited_relations.add(table)
    columns = inspector.get_columns(table)
    for value in relations[table]:
        referred_table = value['referred_table']
        if referred_table == table:
            continue

        index_column = next((i for i, column in enumerate(columns) if column["name"] == value["column"]), None)
        node_data = {"relation": {"source": table, "target": referred_table, "source_column": value['column']}}
        related_table = metadata.tables.get(referred_table)
        if "table_to_many" not in value:
            table_select = select(related_table).where(related_table.c[value["referred_column"]] == data[index_column])
        else:
            join_table = metadata.tables.get(value["table_to_many"])
            table_select = (
                select(related_table).join(join_table).where(join_table.c[value["column"]] == data[index_column])
            )

        data_related = session.execute(table_select).fetchall()
        for row in data_related:
            yield handle_current_data(
                table,
                referred_table,
                node_data,
                data,
                columns,
                row,
                primary_keys,
                inspector.get_columns(referred_table),
            )
            yield from write_parse_relations(
                referred_table, relations, row, primary_keys, session, inspector, metadata, visited_relations
            )


def handle_current_data(
    table: str,
    referred_table: str,
    node_data: dict,
    data: list[tuple],
    columns: list[dict],
    current_data: list[tuple],
    primary_keys: dict[str, list[str]],
    referred_columns: list[dict],
) -> Tree:
    """
    Handle the current data for a table and its referred table.

    :param table: Name of the table.
    :param referred_table: Name of the referred table.
    :param node_data: Dictionary containing relation data.
    :param data: Data for the current table.
    :param columns: List of the table's columns.
    :param current_data: Data for the referred table.
    :param primary_keys: Dictionary of primary keys for tables.
    :param referred_columns: List of the referred table's columns.
    :return: The tree of the relation of the table and the referred_table
    """
    root = Tree(NodeLabel(NodeType.REL, f"{table} -> {referred_table}", node_data), [])
    root.append(write_group_table(table, columns, data, primary_keys.get(table)))
    root.append(
        write_group_table(
            referred_table,
            referred_columns,
            current_data,
            primary_keys.get(referred_table),
        )
    )
    return root

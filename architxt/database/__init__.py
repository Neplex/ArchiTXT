from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from architxt.tree import NodeLabel, NodeType, Tree


def read_database(db_path: str, db_type: str) -> None:
    """
    Read the database, retrieve table information, process table relations, and print the results in a tree format.

    :param db_path: Path to the database.
    :param db_type: Type of the database.
    """
    session = None
    try:
        # Create a connection to the SQLite database
        engine = create_engine(f'{db_type}:///{db_path}')
        inspector = inspect(engine)
        session = sessionmaker(bind=engine)()

        # Retrieve the list of tables in the database
        tables = inspector.get_table_names() or []

        # Get information about the tables (primary keys, foreign keys)
        table_info = read_info_database(tables, inspector)

        # Retrieve primary keys and foreign key relations
        primary_keys = table_info.get("Primary keys", {})
        table_relations = table_info.get("Foreign keys", {})
        parent_tables = get_parent_tables(table_relations, tables)

        final_forest = []
        for table in parent_tables:
            final_forest.extend(process_table_relations(table, table_relations, session, inspector, primary_keys))

        for tree in final_forest:
            print(tree.pformat(margin=255), end="\n\n\n")
            print(tree)

    except SQLAlchemyError as e:
        print(f"SQL Error: {e}")
    finally:
        if session:
            session.close()


def read_info_database(
    tables: list[str],
    inspector: inspect,
) -> dict[str, dict[str, str]]:
    """
    Retrieve information about the database tables, including primary and foreign keys.

    :param tables: List of table names to inspect.
    :param inspector: Database inspection object.
    :return: A dictionary containing primary keys and foreign keys for each table.
    """
    table_info = {"Primary keys": {}, "Foreign keys": {}}

    for table in tables:
        primary_keys = inspector.get_pk_constraint(table)
        table_info["Primary keys"][table] = primary_keys['constrained_columns']

        foreign_keys = inspector.get_foreign_keys(table)
        if foreign_keys:
            table_info["Foreign keys"][table] = [
                {'column': column, 'referred_table': fk['referred_table']}
                for fk in foreign_keys
                for column in fk['constrained_columns']
            ]
    return table_info


def get_parent_tables(relations: dict[str, list[dict[str, str]]], tables: list[str]) -> set[str]:
    """
    Retrieve the parent tables in the database by identifying tables that are not referenced as foreign keys. Handle cycles by including all tables involved.

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
        value = cycle_tables.pop()
        parents.add(value)

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


def process_table_relations(
    table: str,
    relations: dict[str, list[dict[str, str]]],
    session: sessionmaker,
    inspector: inspect,
    primary_keys: dict[str, list[str]],
) -> list[Tree]:
    """
    Process the relations of a given table, retrieve data, and construct tree representations.

    :param table: Name of the table to process.
    :param relations: Dictionary of foreign key relations.
    :param session: SQLAlchemy session object.
    :param inspector: SQLAlchemy inspector object.
    :param primary_keys: Dictionary of primary keys for tables.
    :return: A list of trees representing the relations and data for the table.
    """
    trees = []

    # Retrieve data from the table
    info = session.execute(text(f"SELECT * FROM '{table}'")).fetchmany(3)
    for data in info:
        root = Tree(NodeLabel(NodeType.GROUP, "root"), [])
        root.extend(write_parse_relations(table, relations, data, primary_keys, session, inspector, set()))
        trees.append(root)
    return trees


def write_group_table(
    table: str,
    columns: list[dict],
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

    # Add columns and their respective data to the tree
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
    visited_relations: set[str],
) -> list[Tree]:
    """
    Parse the relations for a table and construct a tree with the related data.

    :param table: Name of the table.
    :param relations: Dictionary of foreign key relations.
    :param data: Data for the current row.
    :param primary_keys: Dictionary of primary keys for tables.
    :param session: SQLAlchemy session object.
    :param inspector: SQLAlchemy inspector object.
    :param visited_relations: Set of visited relations to avoid cycles.
    :return: A list of trees representing the relations and data for the table.
    """
    trees = []
    if table in relations:
        columns = inspector.get_columns(table)
        if table in visited_relations:
            return trees
        visited_relations.add(table)

        for value in relations[table]:
            if value['referred_table'] == table:
                continue
            node_data = {
                "relation": {"source": table, "target": value['referred_table'], "source_column": value['column']}
            }
            root = Tree(NodeLabel(NodeType.REL, f"{table} -> {value['referred_table']}", node_data), [])
            root.append(write_group_table(table, columns, data, primary_keys.get(table)))
            trees.append(root)

            index_column = next(i for i, column in enumerate(columns) if column['name'] == value['column'])
            related_column = next(
                (
                    column
                    for column in inspector.get_columns(value['referred_table'])
                    if column['name'] == value['column']
                ),
                None,
            )
            if not related_column:
                related_column_name = inspector.get_pk_constraint(value['referred_table'])['constrained_columns'][0]
            else:
                related_column_name = related_column['name']

            # Retrieve related data from the referred table
            data_related = session.execute(
                text(f"SELECT * FROM '{value['referred_table']}' WHERE {related_column_name} = '{data[index_column]}'")
            ).fetchone()
            related_tree = write_parse_relations(
                value['referred_table'], relations, data_related, primary_keys, session, inspector, visited_relations
            )
            trees.extend(related_tree)
            root.append(
                write_group_table(
                    value['referred_table'],
                    inspector.get_columns(value['referred_table']),
                    data_related,
                    primary_keys.get(value['referred_table']),
                )
            )
    return trees

from pathlib import Path

import typer
from neo4j import GraphDatabase
from sqlalchemy import create_engine

from architxt.bucket.zodb import ZODBTreeBucket
from architxt.database import export

from .utils import console, load_forest

app = typer.Typer(no_args_is_help=True)


@app.command(name='graph', help="Export the database to Cypher/Bolt compatible database such as Neo4j.")
def export_graph(
    database: list[Path] = typer.Argument(..., help="Path to load the database.", exists=True, readable=True),
    *,
    uri: str = typer.Option(..., help="Database connection string."),
    username: str | None = typer.Option('neo4j', help="Username to use for authentication."),
    password: str | None = typer.Option(None, help="Password to use for authentication."),
) -> None:
    """Export the database as a property graph."""
    auth = (username, password) if username and password else None

    # TODO @neplex: avoid instantiating a bucket here
    # https://github.com/Neplex/ArchiTXT/issues/141
    with (
        ZODBTreeBucket() as forest,
        GraphDatabase.driver(uri, auth=auth) as driver,
        driver.session() as session,
    ):
        forest.update(load_forest(database), commit=True)
        export.export_cypher(forest, session=session)

    console.print('[green]Database exported successfully![/]')


@app.command(name='sql', help="Export the database to SQL compatible database.")
def export_sql(
    database: list[Path] = typer.Argument(..., help="Path to load the database.", exists=True, readable=True),
    *,
    uri: str = typer.Option(..., help="Database connection string."),
) -> None:
    """
    Export the database as a relational database.

    You need to have the necessary driver installed in your environment.
    """
    # TODO @neplex: avoid instantiating a bucket here
    # https://github.com/Neplex/ArchiTXT/issues/141
    with ZODBTreeBucket() as forest, create_engine(uri).connect() as connection:
        forest.update(load_forest(database), commit=True)
        export.export_sql(forest, connection)

    console.print('[green]Database exported successfully![/]')

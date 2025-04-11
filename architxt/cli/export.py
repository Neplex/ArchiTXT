from pathlib import Path

import typer
from neo4j import GraphDatabase

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
    forest = load_forest(database)

    with GraphDatabase.driver(uri, auth=(username, password)) as driver, driver.session() as session:
        export.export_graph(forest, session=session)

    console.print('[green]Database exported successfully![/]')

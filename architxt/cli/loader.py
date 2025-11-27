import functools
from collections.abc import Iterable
from pathlib import Path

import anyio
import click
import mlflow
import typer
from neo4j import GraphDatabase
from sqlalchemy import create_engine

from architxt.bucket import TreeBucket
from architxt.bucket.zodb import ZODBTreeBucket
from architxt.database import loader
from architxt.nlp import raw_load_corpus
from architxt.nlp.entity_resolver import ScispacyResolver
from architxt.nlp.parser.corenlp import CoreNLPParser
from architxt.schema import Schema
from architxt.tree import Tree

from .utils import console, show_schema

__all__ = ['app']

ENTITIES_FILTER = {'TIME', 'MOMENT', 'DUREE', 'DURATION', 'DATE', 'OTHER_ENTITY', 'OTHER_EVENT', 'COREFERENCE'}
RELATIONS_FILTER = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}
ENTITIES_MAPPING = {
    'FREQ': 'FREQUENCY',
    'FREQUENCE': 'FREQUENCY',
    'SIGN_SYMPTOM': 'SOSY',
    'VALEUR': 'VALUE',
    'HEIGHT': 'VALUE',
    'WEIGHT': 'VALUE',
    'MASS': 'VALUE',
    'QUANTITATIVE_CONCEPT': 'VALUE',
    'QUALITATIVE_CONCEPT': 'VALUE',
    'DISTANCE': 'VALUE',
    'VOLUME': 'VALUE',
    'AREA': 'VALUE',
    'LAB_VALUE': 'VALUE',
    'TRAITEMENT': 'THERAPEUTIC_PROCEDURE',
    'MEDICATION': 'THERAPEUTIC_PROCEDURE',
    'DOSE': 'DOSAGE',
    'OUTCOME': 'SOSY',
    'EXAMEN': 'DIAGNOSTIC_PROCEDURE',
    'PATHOLOGIE': 'DISEASE_DISORDER',
    'MODE': 'ADMINISTRATION',
}

app = typer.Typer(no_args_is_help=True)


def _ingest(forest: TreeBucket, trees: Iterable[Tree], incremental: bool) -> None:
    if incremental:
        forest.update(trees, commit=incremental)
    else:
        with forest.transaction():
            forest.update(trees)


async def _async_ingest(forest: TreeBucket, trees: Iterable[Tree], incremental: bool) -> None:
    if incremental:
        await forest.async_update(trees, commit=incremental)
    else:
        with forest.transaction():
            await forest.async_update(trees)


@app.command(name='document', help="Extract document database into a formatted tree.")
def load_document(
    file: Path = typer.Argument(..., exists=True, readable=True, help="The document file to read."),
    *,
    raw: bool = typer.Option(
        False, help="Enable row reading, skipping any transformation to convert it to the metamodel."
    ),
    root_name: str = typer.Option('ROOT', help="The root node name."),
    sample: int | None = typer.Option(None, help="Number of element to sample from the document.", min=1),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    merge_existing: bool = typer.Option(False, help="Should we merge data if output file already exist"),
    incremental: bool = typer.Option(True, help="Enable incremental loading of the database."),
) -> None:
    """Read a parse a document file to a structured tree."""
    if (
        output is not None
        and output.exists()
        and not merge_existing
        and not typer.confirm("The storage path already exists. Merge existing data?")
    ):
        console.print("[red]Cannot store data due to conflict.[/]")
        raise typer.Abort()

    with ZODBTreeBucket(storage_path=output) as forest:
        trees = loader.read_document(file, raw_read=raw, root_name=root_name, sample=sample or 0)
        _ingest(forest, trees, incremental)

        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)


@app.command(name='sql', help="Extract a SQL compatible database into a formatted tree.")
def load_sql(
    uri: str = typer.Argument(..., help="Database connection string."),
    *,
    simplify_association: bool = typer.Option(True, help="Simplify association tables."),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    merge_existing: bool = typer.Option(False, help="Should we merge data if output file already exist"),
    incremental: bool = typer.Option(True, help="Enable incremental loading of the database."),
) -> None:
    """Extract the database schema and relations to a tree format."""
    if (
        output is not None
        and output.exists()
        and not merge_existing
        and not typer.confirm("The storage path already exists. Merge existing data?")
    ):
        console.print("[red]Cannot store data due to conflict.[/]")
        raise typer.Abort()

    with (
        create_engine(uri).connect() as connection,
        ZODBTreeBucket(storage_path=output) as forest,
    ):
        trees = loader.read_sql(connection, simplify_association=simplify_association, sample=sample or 0)
        _ingest(forest, trees, incremental)

        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)


@app.command(name='graph', help="Extract a cypher/bolt compatible database into a formatted tree.")
def load_graph(
    uri: str = typer.Argument(..., help="Database connection string."),
    *,
    username: str | None = typer.Option('neo4j', help="Username to use for authentication."),
    password: str | None = typer.Option(None, help="Password to use for authentication."),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    merge_existing: bool = typer.Option(False, help="Should we merge data if output file already exist"),
    incremental: bool = typer.Option(True, help="Enable incremental loading of the database."),
) -> None:
    if (
        output is not None
        and output.exists()
        and not merge_existing
        and not typer.confirm("The storage path already exists. Merge existing data?")
    ):
        console.print("[red]Cannot store data due to conflict.[/]")
        raise typer.Abort()

    auth = (username, password) if username and password else None

    with (
        GraphDatabase.driver(uri, auth=auth) as driver,
        driver.session() as session,
        ZODBTreeBucket(storage_path=output) as forest,
    ):
        trees = loader.read_cypher(session, sample=sample or 0)
        _ingest(forest, trees, incremental)

        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)


@app.command(name='corpus', help="Extract a database schema form a corpus.", no_args_is_help=True)
def load_corpus(
    corpus_path: list[typer.FileBinaryRead] = typer.Argument(
        ..., exists=True, readable=True, help="Path to the input corpus."
    ),
    *,
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    resolver: str | None = typer.Option(
        None,
        help="The entity resolver to use when loading the corpus.",
        click_type=click.Choice(['umls', 'mesh', 'rxnorm', 'go', 'hpo'], case_sensitive=False),
    ),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    merge_existing: bool = typer.Option(False, help="Should we merge data if output file already exist"),
    incremental: bool = typer.Option(True, help="Enable incremental loading of the database."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
) -> None:
    """Load a corpus and print the database schema as a CFG."""
    if (
        output is not None
        and output.exists()
        and not merge_existing
        and not typer.confirm("The storage path already exists. Merge existing data?")
    ):
        console.print("[red]Cannot store data due to conflict.[/]")
        raise typer.Abort()

    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        mlflow.start_run(description='corpus_processing')

    parser = CoreNLPParser(corenlp_url=corenlp_url)
    _resolver = ScispacyResolver(cleanup=True, translate=True, kb_name=resolver) if resolver else None

    with ZODBTreeBucket(storage_path=output) as forest:
        update = functools.partial(_async_ingest, forest=forest, incremental=incremental)
        anyio.run(
            update,
            raw_load_corpus(
                corpus_path,
                language,
                parser=parser,
                resolver=_resolver,
                cache=cache,
                entities_filter=ENTITIES_FILTER,
                relations_filter=RELATIONS_FILTER,
                entities_mapping=ENTITIES_MAPPING,
                sample=sample,
            ),
        )

        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)

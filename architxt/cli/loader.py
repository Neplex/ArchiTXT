import asyncio
import random
from pathlib import Path

import click
import mlflow
import typer
from rich.panel import Panel
from sqlalchemy import create_engine

from architxt.database.loader import read_database, read_document
from architxt.generator import gen_instance
from architxt.nlp import raw_load_corpus
from architxt.nlp.parser.corenlp import CoreNLPParser
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite

from .utils import console, save_forest, show_metrics, show_schema

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


@app.command(name='document', help="Extract information of a document file into a formatted tree.")
def load_document(
    file: Path = typer.Argument(..., exists=True, readable=True, help="The document file to read."),
    *,
    raw: bool = typer.Option(
        False, help="Enable row reading, skipping any transformation to convert it to the metamodel."
    ),
    root_name: str = typer.Option('ROOT', help="The root node name."),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    output: typer.FileBinaryWrite | None = typer.Option(None, help="Path to save the result."),
) -> None:
    """Read a parse a document file to a structured tree."""
    forest = list(read_document(file, raw_read=raw, root_name=root_name))

    if sample:
        forest = random.sample(forest, sample)

    if output is not None:
        save_forest(forest, output)

    schema = Schema.from_forest(forest, keep_unlabelled=False)
    show_schema(schema)


@app.command(name='database', help="Extract the database information into a formatted tree.")
def load_database(
    db_connection: str = typer.Argument(..., help="Database connection string."),
    *,
    simplify_association: bool = typer.Option(True, help="Simplify association tables."),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    output: typer.FileBinaryWrite | None = typer.Option(None, help="Path to save the result."),
) -> None:
    """Extract the database schema and relations to a tree format."""
    with create_engine(db_connection).connect() as connection:
        forest = list(read_database(connection, simplify_association=simplify_association, sample=sample or 0))

    if output is not None:
        save_forest(forest, output)

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
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting.", min=1),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns.", min=1),
    gen_instances: int = typer.Option(0, help="Number of synthetic instances to generate.", min=0),
    sample: int | None = typer.Option(None, help="Number of sentences to sample from the corpus.", min=1),
    workers: int | None = typer.Option(
        None, help="Number of parallel worker processes to use. Defaults to the number of available CPU cores.", min=1
    ),
    resolver: str | None = typer.Option(
        None,
        help="The entity resolver to use when loading the corpus.",
        click_type=click.Choice(['umls', 'mesh', 'rxnorm', 'go', 'hpo'], case_sensitive=False),
    ),
    output: typer.FileBinaryWrite | None = typer.Option(None, help="Path to save the result."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
    shuffle: bool = typer.Option(False, help="Shuffle the corpus data before processing to introduce randomness."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    metrics: bool = typer.Option(False, help="Show metrics of the simplification."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
) -> None:
    """Automatically structure a corpus as a database instance and print the database schema as a CFG."""
    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        mlflow.start_run(description='corpus_processing')
        mlflow.log_params(
            {
                'has_corpus': True,
                'has_instance': bool(gen_instances),
            }
        )

    try:
        forest = asyncio.run(
            raw_load_corpus(
                corpus_path,
                language,
                parser=CoreNLPParser(corenlp_url=corenlp_url),
                resolver_name=resolver,
                cache=cache,
                entities_filter=ENTITIES_FILTER,
                relations_filter=RELATIONS_FILTER,
                entities_mapping=ENTITIES_MAPPING,
            )
        )
    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error

    if sample:
        if sample < len(forest):
            forest = random.sample(list(forest), sample)
        else:
            console.print(
                "[yellow] You have specified a sample size larger than the total population, "
                "which would result in fewer results than expected."
            )

    # Generate synthetic database instances
    if gen_instances:
        schema = Schema.from_description(
            groups={
                'SOSY': {'SOSY', 'ANATOMIE', 'SUBSTANCE'},
                'TREATMENT': {'SUBSTANCE', 'DOSAGE', 'ADMINISTRATION', 'FREQUENCY'},
                'EXAM': {'DIAGNOSTIC_PROCEDURE', 'ANATOMIE'},
            },
            rels={
                'PRESCRIPTION': ('SOSY', 'TREATMENT'),
                'EXAM_RESULT': ('EXAM', 'SOSY'),
            },
        )
        console.print(Panel(schema.as_cfg(), title="Synthetic Database Schema"))
        with console.status("[cyan]Generating synthetic instances..."):
            forest.extend(gen_instance(schema, size=gen_instances, generate_collections=False))
        console.print(f'[green]Generated {gen_instances} synthetic instances.[/]')

    if shuffle:
        random.shuffle(forest)

    console.print(f'[blue]Rewriting {len(forest)} trees with tau={tau}, epoch={epoch}, min_support={min_support}[/]')
    new_forest = rewrite(forest, tau=tau, epoch=epoch, min_support=min_support, debug=debug, max_workers=workers)

    if output is not None:
        save_forest(new_forest, output)

    # Generate schema
    schema = Schema.from_forest(new_forest, keep_unlabelled=False)
    show_schema(schema)

    if metrics:
        show_metrics(forest, new_forest, schema, tau)

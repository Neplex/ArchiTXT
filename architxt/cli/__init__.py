import asyncio
import subprocess
from collections import Counter
from pathlib import Path

import mlflow
import more_itertools
import typer
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from typer.main import get_command

from architxt.nlp import raw_load_corpus
from architxt.nlp.parser.corenlp import CoreNLPParser
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite

from .export import app as export_app
from .loader import ENTITIES_FILTER, ENTITIES_MAPPING, RELATIONS_FILTER
from .loader import app as loader_app
from .utils import console, load_forest, save_forest, show_metrics, show_schema

app = typer.Typer(
    help="ArchiTXT is a tool for structuring textual data into a valid database model. "
    "It is guided by a meta-grammar and uses an iterative process of tree rewriting.",
    no_args_is_help=True,
)

app.add_typer(loader_app, name="load")
app.add_typer(export_app, name="export")


@app.callback()
def mlflow_setup() -> None:
    mlflow.set_experiment('ArchiTXT')


@app.command(
    help="Launch the web-based UI.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def ui(ctx: typer.Context) -> None:
    """Launch the web-based UI using Streamlit."""
    try:
        from architxt import ui

        subprocess.run(['streamlit', 'run', ui.__file__, *ctx.args], check=True)

    except FileNotFoundError as error:
        console.print(
            "[red]Streamlit is not installed or not found. Please install it with `pip install architxt[ui]` to use the UI.[/]"
        )
        raise typer.Exit(code=1) from error


@app.command(help="Simplify a bunch of databased together.", no_args_is_help=True)
def simplify(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting.", min=1),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns.", min=1),
    sample: int | None = typer.Option(None, help="Number of tree to use from the simplification.", min=1),
    workers: int | None = typer.Option(
        None, help="Number of parallel worker processes to use. Defaults to the number of available CPU cores.", min=1
    ),
    output: typer.FileBinaryWrite | None = typer.Option(None, help="Path to save the result."),
    shuffle: bool = typer.Option(False, help="Shuffle the data before processing to introduce randomness."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    metrics: bool = typer.Option(False, help="Show metrics of the simplification."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
) -> None:
    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        mlflow.start_run(description='simplification')
        for file in files:
            mlflow.log_input(MetaDataset(CodeDatasetSource({}), name=file.name))

    forest = load_forest(files, sample=sample, shuffle=shuffle)

    console.print(f'[blue]Rewriting {len(forest)} trees with tau={tau}, epoch={epoch}, min_support={min_support}[/]')
    new_forest = rewrite(forest, tau=tau, epoch=epoch, min_support=min_support, debug=debug, max_workers=workers)

    if output:
        save_forest(new_forest, output)

    # Generate schema
    schema = Schema.from_forest(new_forest, keep_unlabelled=False)
    show_schema(schema)

    if metrics:
        show_metrics(forest, new_forest, schema, tau)


@app.command(help="Display overall statistics for the corpus.")
def corpus_stats(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    *,
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
) -> None:
    """Display overall corpus statistics."""
    forest = asyncio.run(
        raw_load_corpus(
            corpus_path,
            language,
            parser=CoreNLPParser(corenlp_url=corenlp_url),
            cache=cache,
            entities_filter=ENTITIES_FILTER,
            relations_filter=RELATIONS_FILTER,
            entities_mapping=ENTITIES_MAPPING,
        )
    )

    # Entity Count
    entity_count = Counter([ent.label().name for tree in forest for ent in tree.entities()])

    tables = []
    for chunk in more_itertools.chunked_even(entity_count.most_common(), 10):
        entity_table = Table()
        entity_table.add_column("Entity", style="cyan", no_wrap=True)
        entity_table.add_column("Count", style="magenta")

        for entity, count in chunk:
            entity_table.add_row(entity, str(count))

        tables.append(entity_table)

    # Compute statistics
    total_trees = len(forest)
    total_entities = sum(len(tree.entities()) for tree in forest)
    tree_heights = [tree.height() for tree in forest]
    tree_sizes = [len(tree.leaves()) for tree in forest]
    avg_height = sum(tree_heights) / len(tree_heights) if tree_heights else 0
    max_height = max(tree_heights, default=0)
    avg_size = sum(tree_sizes) / len(tree_sizes) if tree_sizes else 0
    max_size = max(tree_sizes, default=0)

    stats_table = Table()
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="magenta")

    stats_table.add_row("Total Trees", str(total_trees))
    stats_table.add_row("Total Entities", str(total_entities))
    stats_table.add_row("Average Tree Height", f"{avg_height:.3f}")
    stats_table.add_row("Maximum Tree Height", str(max_height))
    stats_table.add_row("Average Tree size", f"{avg_size:.3f}")
    stats_table.add_row("Maximum Tree size", str(max_size))

    console.print(Columns([*tables, stats_table], equal=True))


@app.command(help="Display details about the largest tree in the corpus.")
def largest_tree(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    *,
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
) -> None:
    """Display the largest tree in the corpus along with its sentence and structure."""
    forest = asyncio.run(
        raw_load_corpus(
            corpus_path,
            language,
            parser=CoreNLPParser(corenlp_url=corenlp_url),
            cache=cache,
            entities_filter=ENTITIES_FILTER,
            relations_filter=RELATIONS_FILTER,
            entities_mapping=ENTITIES_MAPPING,
        )
    )

    # Find the largest tree
    tree = max(forest, key=lambda t: len(t.leaves()), default=None)

    if tree:
        sentence = " ".join(tree.leaves())
        tree_display = tree.pformat(margin=255)

        console.print(Panel(sentence, title="Sentence"))
        console.print(Panel(tree_display, title="Tree"))

    else:
        console.print("[yellow]No trees found in the corpus.[/]")


# Click command used for Sphinx documentation
_click_command = get_command(app)

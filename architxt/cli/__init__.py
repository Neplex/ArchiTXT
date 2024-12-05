import hashlib
import subprocess
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory
from typing import BinaryIO

import cloudpickle
import mlflow
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from architxt.algo import rewrite
from architxt.db import Schema
from architxt.generator import gen_instance
from architxt.nlp import get_enriched_forest, get_sentence_from_disk
from architxt.tree import Tree

console = Console()


def load_or_cache_corpus(
    archive_file: BytesIO | BinaryIO,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    corenlp_url: str,
    language: str,
) -> list[Tree]:
    """
    Load the corpus from disk or cache.

    :param archive_file: An in-memory file object representing the corpus archive.
    :param entities_filter: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param entities_mapping: A dictionary mapping entity names to new values. If None, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :param corenlp_url: The URL of the CoreNLP server.
    :param language: The language to use for parsing.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    try:
        # Generate a cache key based on the archive file's content
        file_hash = hashlib.file_digest(archive_file, hashlib.md5)

        if entities_filter:
            file_hash.update('E'.join(sorted(entities_filter)).encode())
        if relations_filter:
            file_hash.update('R'.join(sorted(relations_filter)).encode())

        key = file_hash.hexdigest()
        corpus_cache_path = Path(f'{key}.pkl')

        # Attempt to load from cache if available
        if corpus_cache_path.exists():
            console.print(f'[green]Loading corpus from cache:[/] {corpus_cache_path.absolute()}')
            with corpus_cache_path.open('rb') as cache_file:
                return cloudpickle.load(cache_file)

        archive_file.seek(0)

        # If the cache does not exist, process the archive
        with (
            console.status(f'[yellow]Loading corpus from disk:[/] {archive_file.name}'),
            TarFile.open(fileobj=archive_file) as corpus,
            TemporaryDirectory() as tmp_dir,
        ):
            # Extract archive contents to a temporary directory
            corpus.extractall(tmp_dir)
            tmp_path = Path(tmp_dir)

            # Parse sentences and enrich the forest
            sentences = get_sentence_from_disk(
                tmp_path,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                relations_mapping=relations_mapping,
            )
            forest = list(get_enriched_forest(sentences, corenlp_url=corenlp_url, language=language))
            console.print(f'[green]Dataset loaded! {len(forest)} sentences found.[/]')

        # Save processed data to cache
        console.print(f'[blue]Saving cache file to:[/] {corpus_cache_path.absolute()}')
        with corpus_cache_path.open('wb') as cache_file:
            cloudpickle.dump(forest, cache_file)

        return forest

    except Exception as e:
        console.print(f'[red]Error while processing corpus:[/] {e}')
        raise


def cli_run(
    corpus_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold."),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting."),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    gen_instances: int = typer.Option(0, help="Number of synthetic instances to generate."),
    language: str = typer.Option('French', help="Language of the input corpus."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    workers: int | None = typer.Option(
        None, help="Number of parallel worker processes to use. Defaults to the number of available CPU cores."
    ),
) -> None:
    """
    Automatically structure a corpus as a database instance and print the database schema as a CFG.
    """
    entities_filter = {'MOMENT', 'DUREE', 'DATE'}
    relations_filter = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}
    entities_mapping = {'FREQ': 'FREQUENCE'}

    # Load the corpus
    try:
        with corpus_path.open('rb') as corpus:
            forest = load_or_cache_corpus(
                corpus,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                corenlp_url=corenlp_url,
                language=language,
            )

    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error

    # Generate synthetic database instances
    if gen_instances:
        with console.status("[cyan]Generating synthetic instances..."):
            gen_trees = gen_instance(
                groups={
                    'SOSY': ('SOSY', 'ANATOMIE', 'SUBSTANCE'),
                    'TREATMENT': ('SUBSTANCE', 'DOSE', 'MODE', 'FREQUENCE'),
                    'EXAM': ('EXAMEN', 'ANATOMIE'),
                },
                rels={
                    'PRESCRIPTION': ('SOSY', 'TREATMENT'),
                    'EXAM': ('EXAM', 'SOSY'),
                },
                size=gen_instances,
            )
            forest.extend(gen_trees)

    # Rewrite the trees
    mlflow.enable_system_metrics_logging()
    mlflow.log_params(
        {
            'has_corpus': True,
            'has_instance': bool(gen_instances),
        }
    )

    console.print(f'[blue]Rewriting trees with tau={tau}, epoch={epoch}, min_support={min_support}[/]')
    forest = rewrite(forest, tau=tau, epoch=epoch, min_support=min_support, debug=debug, max_workers=workers)

    # Generate schema
    schema = Schema.from_forest(forest, keep_unlabelled=False)
    schema_str = schema.as_cfg()
    mlflow.log_text(schema_str, 'schema.txt')

    console.print(
        Panel(
            schema_str,
            title="Schema as CFG",
            subtitle='[green]Valid Schema[/]' if schema.verify() else '[red]Invalid Schema[/]',
        )
    )


def cli_ui() -> None:
    """
    Launch the web-based UI using Streamlit.
    """
    try:
        from architxt import ui

        subprocess.run(['streamlit', 'run', ui.__file__], check=True)

    except FileNotFoundError as error:
        console.print(
            "[red]Streamlit is not installed or not found. Please install it with `pip install architxt[ui]` to use the UI.[/]"
        )
        raise typer.Exit(code=1) from error


def cli_stats(
    corpus_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    language: str = typer.Option('French', help="Language of the input corpus."),
) -> None:
    """
    Display overall corpus statistics.
    """
    entities_filter = {'MOMENT', 'DUREE', 'DATE'}
    relations_filter = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}
    entities_mapping = {'FREQ': 'FREQUENCE'}

    # Load the corpus
    try:
        with corpus_path.open('rb') as corpus:
            forest = load_or_cache_corpus(
                corpus,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                corenlp_url=corenlp_url,
                language=language,
            )

    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error

    forest = list(filter(lambda x: len(x.leaves()) < 50, forest))

    # Compute statistics
    total_trees = len(forest)
    total_entities = sum(len(tree.entities()) for tree in forest)
    tree_heights = [tree.height() for tree in forest]
    tree_sizes = [len(tree.leaves()) for tree in forest]
    avg_height = sum(tree_heights) / len(tree_heights) if tree_heights else 0
    max_height = max(tree_heights, default=0)
    avg_size = sum(tree_sizes) / len(tree_sizes) if tree_sizes else 0
    max_size = max(tree_sizes, default=0)

    # Create a statistics table
    table = Table()
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total Trees", str(total_trees))
    table.add_row("Total Entities", str(total_entities))
    table.add_row("Average Tree Height", f"{avg_height:.2f}")
    table.add_row("Maximum Tree Height", str(max_height))
    table.add_row("Average Tree size", f"{avg_size:.2f}")
    table.add_row("Maximum Tree size", str(max_size))

    console.print(table)


def cli_largest_tree(
    corpus_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    language: str = typer.Option('French', help="Language of the input corpus."),
) -> None:
    """
    Display the largest tree in the corpus along with its sentence and structure.
    """
    entities_filter = {'MOMENT', 'DUREE', 'DATE'}
    relations_filter = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}
    entities_mapping = {'FREQ': 'FREQUENCE'}

    # Load the corpus
    try:
        with corpus_path.open('rb') as corpus:
            forest = load_or_cache_corpus(
                corpus,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                corenlp_url=corenlp_url,
                language=language,
            )

    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error

    forest = list(filter(lambda x: len(x.leaves()) < 50, forest))

    # Find the largest tree
    largest_tree = max(forest, key=lambda t: len(t.leaves()), default=None)

    if largest_tree:
        sentence = " ".join(largest_tree.leaves())
        largest_tree_display = largest_tree.pformat(margin=255)

        console.print(
            Panel(
                sentence,
                title="Sentence",
            )
        )
        console.print(
            Panel(
                largest_tree_display,
                title="Tree",
            )
        )

    else:
        console.print("[yellow]No trees found in the corpus.[/]")


def main() -> None:
    """
    Main entry point for the CLI.
    """
    mlflow.set_experiment('ArchiTXT')

    app = typer.Typer(
        help="ArchiTXT is a tool for structuring textual data into a valid database model. "
        "It is guided by a meta-grammar and uses an iterative process of tree rewriting."
    )
    app.command('run', help="Extract a database schema form a corpus.")(cli_run)
    app.command('ui', help="Launch the web-based UI.")(cli_ui)
    app.command('stats', help="Display overall statistics for the corpus.")(cli_stats)
    app.command('largest-tree', help="Display details about the largest tree in the corpus.")(cli_largest_tree)

    app()

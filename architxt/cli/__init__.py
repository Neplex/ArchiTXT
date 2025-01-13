import hashlib
import random
import subprocess
from collections import Counter
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory
from typing import BinaryIO

import cloudpickle
import mlflow
import more_itertools
import typer
from mlflow.data import Dataset
from mlflow.data.code_dataset_source import CodeDatasetSource
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from architxt.generator import gen_instance
from architxt.nlp import get_enriched_forest, get_sentence_from_disk
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite
from architxt.tree import Tree

console = Console()

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


def load_or_cache_corpus(
    archive_file: BytesIO | BinaryIO,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    corenlp_url: str,
    language: str,
    name: str | None = None,
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
    :param name: The corpus name.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    try:
        # Generate a cache key based on the archive file's content
        archive_file.seek(0)
        file_hash = hashlib.file_digest(archive_file, hashlib.md5)

        if entities_filter:
            file_hash.update('$E'.join(sorted(entities_filter)).encode())
        if relations_filter:
            file_hash.update('$R'.join(sorted(relations_filter)).encode())
        if entities_mapping:
            file_hash.update('$EM'.join(sorted(f'{key}={value}' for key, value in entities_mapping.items())).encode())
        if relations_mapping:
            file_hash.update('$RM'.join(sorted(f'{key}={value}' for key, value in relations_mapping.items())).encode())

        key = file_hash.hexdigest()
        corpus_cache_path = Path(f'{key}.pkl')

        mlflow.log_input(
            Dataset(
                CodeDatasetSource(
                    {
                        'entities_filter': sorted(entities_filter),
                        'relations_filter': sorted(relations_filter),
                        'entities_mapping': entities_mapping,
                        'relations_mapping': relations_mapping,
                        'cache_file': str(corpus_cache_path.absolute()),
                    }
                ),
                name=name,
                digest=key,
            )
        )

        # Attempt to load from cache if available
        if corpus_cache_path.exists():
            console.print(f'[green]Loading corpus from cache:[/] {corpus_cache_path.absolute()}')
            with corpus_cache_path.open('rb') as cache_file:
                return cloudpickle.load(cache_file)

        archive_file.seek(0)
        console.print(f'[yellow]Loading corpus from disk:[/] {archive_file.name}')

        # If the cache does not exist, process the archive
        with (
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


def load_corpus_paths(
    corpus_path: list[Path],
    language: list[str],
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    corenlp_url: str,
) -> list[Tree]:
    try:
        forest = []
        for path, lang in zip(corpus_path, language, strict=True):
            with path.open('rb') as corpus:
                forest += load_or_cache_corpus(
                    corpus,
                    entities_filter=entities_filter,
                    relations_filter=relations_filter,
                    entities_mapping=entities_mapping,
                    relations_mapping=relations_mapping,
                    corenlp_url=corenlp_url,
                    language=lang,
                    name=path.name,
                )

        return forest

    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error


def cli_run(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    *,
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    tau: float = typer.Option(0.7, help="The similarity threshold."),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting."),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns."),
    gen_instances: int = typer.Option(0, help="Number of synthetic instances to generate."),
    sample: int = typer.Option(0, help="Number of sentences to sample from the corpus."),
    shuffle: bool = typer.Option(False, help="Shuffle the corpus data before processing to introduce randomness."),
    workers: int | None = typer.Option(
        None, help="Number of parallel worker processes to use. Defaults to the number of available CPU cores."
    ),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    output: Path | None = typer.Option(None, exists=False, writable=True, help="Path to save the result."),
) -> None:
    """
    Automatically structure a corpus as a database instance and print the database schema as a CFG.
    """
    forest = load_corpus_paths(
        corpus_path,
        language,
        corenlp_url=corenlp_url,
        entities_filter=ENTITIES_FILTER,
        relations_filter=RELATIONS_FILTER,
        entities_mapping=ENTITIES_MAPPING,
    )
    forest = list(filter(lambda t: len(t.leaves()) < 30, forest))

    # Rewrite the trees
    mlflow.enable_system_metrics_logging()
    mlflow.log_params(
        {
            'has_corpus': True,
            'has_instance': bool(gen_instances),
        }
    )

    if sample:
        forest = random.sample(forest, sample)

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
            forest.extend(gen_instance(schema, size=gen_instances))
        console.print(f'[green]Generated {gen_instances} synthetic instances.[/]')

    if shuffle:
        random.shuffle(forest)

    console.print(f'[blue]Rewriting {len(forest)} trees with tau={tau}, epoch={epoch}, min_support={min_support}[/]')
    new_forest = rewrite(forest, tau=tau, epoch=epoch, min_support=min_support, debug=debug, max_workers=workers)

    if output:
        with console.status(f"[cyan]Saving instance to {output}..."), output.open('wb') as output_file:
            cloudpickle.dump(new_forest, output_file)

    # Generate schema
    schema = Schema.from_forest(new_forest, keep_unlabelled=False)
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
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
) -> None:
    """
    Display overall corpus statistics.
    """
    forest = load_corpus_paths(
        corpus_path,
        language,
        corenlp_url=corenlp_url,
        entities_filter=ENTITIES_FILTER,
        relations_filter=RELATIONS_FILTER,
        entities_mapping=ENTITIES_MAPPING,
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
    stats_table.add_row("Average Tree Height", f"{avg_height:.2f}")
    stats_table.add_row("Maximum Tree Height", str(max_height))
    stats_table.add_row("Average Tree size", f"{avg_size:.2f}")
    stats_table.add_row("Maximum Tree size", str(max_size))

    console.print(Columns([*tables, stats_table], equal=True))


def cli_largest_tree(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
) -> None:
    """
    Display the largest tree in the corpus along with its sentence and structure.
    """
    forest = load_corpus_paths(
        corpus_path,
        language,
        corenlp_url=corenlp_url,
        entities_filter=ENTITIES_FILTER,
        relations_filter=RELATIONS_FILTER,
        entities_mapping=ENTITIES_MAPPING,
    )

    # Find the largest tree
    largest_tree = max(forest, key=lambda t: len(t.leaves()), default=None)

    if largest_tree:
        sentence = " ".join(largest_tree.leaves())
        largest_tree_display = largest_tree.pformat(margin=255)

        console.print(Panel(sentence, title="Sentence"))
        console.print(Panel(largest_tree_display, title="Tree"))

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

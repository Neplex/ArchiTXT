import asyncio
import hashlib
import random
import subprocess
from collections import Counter
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory
from typing import BinaryIO

import click
import cloudpickle
import mlflow
import more_itertools
import typer
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.main import get_command

from architxt.generator import gen_instance
from architxt.metrics import Metrics
from architxt.nlp.brat import load_brat_dataset
from architxt.nlp.entity_resolver import EntityResolver, ScispacyResolver
from architxt.nlp.parser import Parser
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite
from architxt.tree import Forest

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

console = Console()
app = typer.Typer(
    help="ArchiTXT is a tool for structuring textual data into a valid database model. "
    "It is guided by a meta-grammar and uses an iterative process of tree rewriting.",
)


async def write_cache(forest: Forest, path: Path) -> None:
    with path.open('wb') as cache_file:
        await asyncio.to_thread(cloudpickle.dump, forest, cache_file, protocol=5, buffer_callback=None)


async def read_cache(path: Path) -> Forest:
    with path.open('rb') as cache_file:
        return await asyncio.to_thread(cloudpickle.load, cache_file)


async def load_or_cache_corpus(
    archive_file: BytesIO | BinaryIO,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    parser: Parser,
    language: str,
    name: str | None = None,
    resolver: EntityResolver | None = None,
    cache: bool = True,
) -> Forest:
    """
    Load the corpus from disk or cache.

    :param archive_file: An in-memory file object representing the corpus archive.
    :param entities_filter: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param entities_mapping: A dictionary mapping entity names to new values. If None, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :param parser: The NLP parser to use.
    :param language: The language to use for parsing.
    :param name: The corpus name.
    :param resolver: An optional entity resolver to use.
    :param cache: Whether to cache the computed forest or not.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    try:
        # Generate a cache key based on the archive file's content
        archive_file.seek(0)
        file_hash = await asyncio.to_thread(hashlib.file_digest, archive_file, hashlib.md5)
        file_hash.update(language.encode())

        if entities_filter:
            file_hash.update('$E'.join(sorted(entities_filter)).encode())
        if relations_filter:
            file_hash.update('$R'.join(sorted(relations_filter)).encode())
        if entities_mapping:
            file_hash.update('$EM'.join(sorted(f'{key}={value}' for key, value in entities_mapping.items())).encode())
        if relations_mapping:
            file_hash.update('$RM'.join(sorted(f'{key}={value}' for key, value in relations_mapping.items())).encode())
        if resolver:
            file_hash.update(resolver.name.encode())

        key = file_hash.hexdigest()
        corpus_cache_path = Path(f'{key}.pkl')

        mlflow.log_input(
            MetaDataset(
                CodeDatasetSource(
                    {
                        'entities_filter': sorted(entities_filter or []),
                        'relations_filter': sorted(relations_filter or []),
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
        if cache and corpus_cache_path.exists():
            console.print(f'[green]Loading corpus from cache:[/] {corpus_cache_path.absolute()}')
            return await read_cache(corpus_cache_path)

        archive_file.seek(0)
        console.print(f'[yellow]Loading corpus from disk:[/] {archive_file.name}')

        # If the cache does not exist, process the archive
        with (
            TarFile.open(fileobj=archive_file) as corpus,
            TemporaryDirectory() as tmp_dir,
        ):
            # Extract archive contents to a temporary directory
            await asyncio.to_thread(corpus.extractall, tmp_dir, None)
            tmp_path = Path(tmp_dir)

            # Parse sentences and enrich the forest
            sentences = load_brat_dataset(
                tmp_path,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                relations_mapping=relations_mapping,
            )
            forest = [tree async for _, tree in parser.parse_batch(sentences, language=language, resolver=resolver)]
            console.print(f'[green]Dataset loaded! {len(forest)} sentences found.[/]')

        # Save processed data to cache
        if cache:
            console.print(f'[blue]Saving cache file to:[/] {corpus_cache_path.absolute()}')
            await write_cache(forest, corpus_cache_path)

    except Exception as e:
        console.print(f'[red]Error while processing corpus:[/] {e}')
        raise

    else:
        return forest


async def load_corpus(
    archive_path: Path,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    language: str,
    parser: Parser,
    resolver: EntityResolver | None = None,
    cache: bool = True,
) -> Forest:
    with archive_path.open('rb') as corpus:
        return await load_or_cache_corpus(
            corpus,
            name=archive_path.name,
            entities_filter=entities_filter,
            relations_filter=relations_filter,
            entities_mapping=entities_mapping,
            relations_mapping=relations_mapping,
            language=language,
            parser=parser,
            resolver=resolver,
            cache=cache,
        )


async def load_corpus_batch(
    corpus_path: list[Path],
    language: list[str],
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    corenlp_url: str,
    resolver_name: str | None = None,
    cache: bool = True,
) -> Forest:
    try:
        with Parser(corenlp_url=corenlp_url) as parser:
            resolver_ctx = (
                ScispacyResolver(cleanup=True, translate=True, kb_name=resolver_name)
                if resolver_name
                else nullcontext()
            )

            async with resolver_ctx as resolver:
                forests = await asyncio.gather(
                    *[
                        load_corpus(
                            path,
                            entities_filter=entities_filter,
                            relations_filter=relations_filter,
                            entities_mapping=entities_mapping,
                            relations_mapping=relations_mapping,
                            parser=parser,
                            language=language,
                            resolver=resolver,
                            cache=cache,
                        )
                        for path, language in zip(corpus_path, language, strict=True)
                    ]
                )

                return [tree for forest in forests for tree in forest]

    except Exception as error:
        console.print_exception()
        raise typer.Exit(code=1) from error


@app.command(help="Extract a database schema form a corpus.", no_args_is_help=True)
def run(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
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
    output: Path | None = typer.Option(None, exists=False, writable=True, help="Path to save the result."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
    shuffle: bool = typer.Option(False, help="Shuffle the corpus data before processing to introduce randomness."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
) -> None:
    """Automatically structure a corpus as a database instance and print the database schema as a CFG."""
    forest = asyncio.run(
        load_corpus_batch(
            corpus_path,
            language,
            corenlp_url=corenlp_url,
            resolver_name=resolver,
            cache=cache,
            entities_filter=ENTITIES_FILTER,
            relations_filter=RELATIONS_FILTER,
            entities_mapping=ENTITIES_MAPPING,
        )
    )

    # Rewrite the trees
    mlflow.enable_system_metrics_logging()
    mlflow.log_params(
        {
            'has_corpus': True,
            'has_instance': bool(gen_instances),
        }
    )

    if sample:
        if sample < len(forest):
            forest = random.sample(list(forest), sample)
        else:
            console.print(
                "[yellow] You have specified a sample size larger than the total population, "
                "which may result in fewer results than expected."
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
            title="Schema as CFG (labelled nodes only)",
            subtitle='[green]Valid Schema[/]' if schema.verify() else '[red]Invalid Schema[/]',
        )
    )

    with console.status("[cyan]Computing metrics. This may take a while. Please wait..."):
        valid_instance = schema.extract_valid_trees(new_forest)
        metrics = Metrics(forest, valid_instance)

        metrics_table = Table("Metric", "Value", title="Valid instance")

        metrics_table.add_row("Coverage ▲", f"{metrics.coverage():.3f}")
        metrics_table.add_row("Similarity ▲", f"{metrics.similarity():.3f}")
        metrics_table.add_row("Edit distance ▼", str(metrics.edit_distance()))
        metrics_table.add_row("Redundancy (1.0) ▼", f"{metrics.redundancy(tau=1.0):.3f}")
        metrics_table.add_row("Redundancy (0.7) ▼", f"{metrics.redundancy(tau=0.7):.3f}")
        metrics_table.add_row("Redundancy (0.5) ▼", f"{metrics.redundancy(tau=0.5):.3f}")

        metrics_table.add_section()

        metrics_table.add_row("Cluster Mutual Information ▲", f"{metrics.cluster_ami(tau=tau):.3f}")
        metrics_table.add_row("Cluster Completeness ▲", f"{metrics.cluster_completeness(tau=tau):.3f}")

        schema_old = Schema.from_forest(forest, keep_unlabelled=True)
        grammar_metrics_table = Table("Metric", "Before Value", "After Value", title="Schema grammar")
        grammar_metrics_table.add_row(
            "Productions ▼",
            str(len(schema_old.productions())),
            f"{len(schema.productions())} ({len(schema.productions()) / len(schema_old.productions()) * 100:.3f}%)",
        )
        grammar_metrics_table.add_row("Overlap ▼", f"{schema_old.group_overlap:.3f}", f"{schema.group_overlap:.3f}")
        grammar_metrics_table.add_row(
            "Balance ▲", f"{schema_old.group_balance_score:.3f}", f"{schema.group_balance_score:.3f}"
        )

        console.print(Columns([metrics_table, grammar_metrics_table]))


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


@app.command(help="Display overall statistics for the corpus.")
def stats(
    corpus_path: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path to the input corpus."),
    language: list[str] = typer.Option(['French'], help="Language of the input corpus."),
    *,
    corenlp_url: str = typer.Option('http://localhost:9000', help="URL of the CoreNLP server."),
    cache: bool = typer.Option(True, help="Enable caching of the analyzed corpus to prevent re-parsing."),
) -> None:
    """Display overall corpus statistics."""
    forest = asyncio.run(
        load_corpus_batch(
            corpus_path,
            language,
            corenlp_url=corenlp_url,
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
        load_corpus_batch(
            corpus_path,
            language,
            corenlp_url=corenlp_url,
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

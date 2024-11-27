import hashlib
import subprocess
from pathlib import Path

import mlflow
import typer
from ray import cloudpickle
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner

from architxt.algo import rewrite
from architxt.db import Schema
from architxt.generator import gen_instance
from architxt.nlp import get_enriched_forest, get_sentence_from_disk
from architxt.tree import Tree

console = Console()


def load_or_cache_corpus(
    corpus_path: Path,
    *,
    entities_filter: set[str],
    relations_filter: set[str],
    entities_mapping: dict[str, str],
    corenlp_url: str,
    language: str,
) -> list[Tree]:
    """Load the corpus from disk or cache."""
    key = hashlib.md5(
        (corpus_path.name + 'E'.join(sorted(entities_filter)) + 'R'.join(sorted(relations_filter))).encode()
    ).hexdigest()
    corpus_cache_path = Path(f'{key}.pkl')

    if corpus_cache_path.exists():
        console.print(f'[green]Loading corpus from cache:[/] {corpus_cache_path.absolute()}')
        with open(corpus_cache_path, 'rb') as cache_file:
            return cloudpickle.load(cache_file)

    with Spinner(f'[yellow]Loading corpus from disk:[/] {corpus_path.absolute()}'):
        sentences = get_sentence_from_disk(
            corpus_path,
            entities_filter=entities_filter,
            relations_filter=relations_filter,
            entities_mapping=entities_mapping,
        )
        forest = list(get_enriched_forest(sentences, corenlp_url=corenlp_url, language=language))

    console.print(f'[blue]Saving cache file to:[/] {corpus_cache_path.absolute()}')
    with open(corpus_cache_path, 'wb') as cache_file:
        cloudpickle.dump(forest, cache_file)

    return forest


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
) -> None:
    """
    Automatically structure a corpus as a database instance and print the database schema as a CFG.
    """
    mlflow.log_params(
        {
            'has_corpus': True,
            'has_instance': bool(gen_instances),
        }
    )

    entities_filter = {'MOMENT', 'DUREE', 'DATE'}
    relations_filter = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}
    entities_mapping = {'FREQ': 'FREQUENCE'}

    # Load the corpus
    try:
        forest = load_or_cache_corpus(
            corpus_path,
            entities_filter=entities_filter,
            relations_filter=relations_filter,
            entities_mapping=entities_mapping,
            corenlp_url=corenlp_url,
            language=language,
        )
        console.print(f'[green]Dataset loaded! {len(forest)} sentences found.[/]')

    except Exception as e:
        console.print(f"[red]Error loading corpus:[/] {e}")
        raise typer.Exit(code=1)

    # Generate synthetic database instances
    if gen_instances:
        with Spinner("[cyan]Generating synthetic instances..."):
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
    console.print(f'[blue]Rewriting trees with tau={tau}, epoch={epoch}, min_support={min_support}[/]')
    forest = rewrite(forest, tau=tau, epoch=epoch, min_support=min_support, debug=debug)

    # Generate schema
    schema = Schema.from_forest(forest)
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
        from .. import ui

        subprocess.run(['streamlit', 'run', ui.__file__], check=True)

    except FileNotFoundError:
        console.print(
            "[red]Streamlit is not installed or not found. Please install it with `pip install architxt[ui]` to use the UI.[/]"
        )
        raise typer.Exit(code=1)


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
    app()

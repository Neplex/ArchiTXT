import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import cloudpickle
import mlflow
import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from architxt.metrics import Metrics
from architxt.schema import Schema
from architxt.tree import Forest

__all__ = ['console', 'load_forest', 'save_forest', 'show_metrics', 'show_schema']


console = Console()


def show_schema(schema: Schema) -> None:
    schema_str = schema.as_cfg()
    mlflow.log_text(schema_str, 'schema.txt')

    console.print(
        Panel(
            schema_str,
            title="Schema as CFG (labelled nodes only)",
            subtitle='[green]Valid Schema[/]' if schema.verify() else '[red]Invalid Schema[/]',
        )
    )


def show_metrics(forest: Forest, new_forest: Forest, schema: Schema, tau: float) -> None:
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


def save_forest(forest: Forest, output: BytesIO | BinaryIO) -> None:
    """
    Serialize and save the forest object to a buffer.

    :param forest: The forest object to be serialized and saved.
    :param output: The buffer or file-like object where the forest  will be saved.

    >>> with open('forest.pkl', 'wb') as f: # doctest: +SKIP
    ...     save_forest(forest, f)
    """
    with console.status(f"[cyan]Saving instance to {typer.format_filename(output.name)}..."):
        cloudpickle.dump(forest, output)


def load_forest(files: Iterable[str | Path], *, sample: int = 0, shuffle: bool = False) -> Forest:
    """
    Load a forest from a list of binary files.

    :param files: List of file paths to read and deserialize into a forest.
    :param sample: The number of trees to sample from the forest. If 0, the entire forest is loaded.
    :param shuffle: Whether to shuffle the forest after loading.

    :returns: A list containing the deserialized forest data.

    >>> forest = load_forest(['forest1.pkl', 'forest2.pkl'], sample=100, shuffle=True) # doctest: +SKIP
    """
    forest = []

    with Progress() as progress:
        for path in files:
            with progress.open(path, 'rb', description=f'Reading {path}...') as file:
                forest.extend(cloudpickle.load(file))

    if sample:
        if sample < len(forest):
            forest = random.sample(list(forest), sample)
        else:
            console.print(
                "[yellow] You have specified a sample size larger than the total population, "
                "which would result in fewer results than expected."
            )

    if shuffle:
        random.shuffle(forest)

    return forest

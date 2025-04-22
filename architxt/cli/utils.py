from collections.abc import Generator, Iterable
from pathlib import Path

import mlflow
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from architxt.metrics import Metrics
from architxt.schema import Schema
from architxt.tree import Forest, Tree, ZODBTreeBucket

__all__ = ['console', 'load_forest', 'show_metrics', 'show_schema']


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


def load_forest(files: Iterable[str | Path]) -> Generator[Tree, None, None]:
    """
    Load a forest from a list of zodb files.

    :param files: List of file paths to read into a forest.
    :yield: Trees from the list of data files.

    >>> forest = load_forest(['forest1.data', 'forest2.data']) # doctest: +SKIP
    """
    with Progress() as progress:
        task_ids = [progress.add_task(f'Reading {file_path}...', start=False) for file_path in files]

        for file_path, task_id in zip(files, task_ids):
            progress.start_task(task_id)

            with ZODBTreeBucket(storage_path=file_path, read_only=True) as forest:
                for tree in progress.track(forest, task_id=task_id):
                    yield tree.copy()

import gc
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import typer
from architxt.bucket.zodb import ZODBTreeBucket
from architxt.cli.utils import load_forest
from architxt.similarity import DECAY
from architxt.simplification.tree_rewriting import rewrite
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from rich.console import Console

console = Console()

app = typer.Typer(help="ArchiTXT Benchmarking suite.", no_args_is_help=True)


@app.callback()
def mlflow_setup() -> None:
    mlflow.set_experiment('ArchiTXT-Benchmark')


@app.command(help="Benchmark the simplification algorithm of Architxt.")
def simplify(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting.", min=1),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns.", min=1),
    workers: list[int] = typer.Option(
        default=(1,),
        help="Number of parallel worker processes to use for each run. Defaults to the number of available CPU cores.",
    ),
    sizes: list[int] = typer.Option(default=(100,), help="Number of instances to generate for each size."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
    log_system_metrics: bool = typer.Option(False, help="Enable logging of system metrics to MLFlow."),
) -> None:
    run_ctx = nullcontext()

    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        run_ctx = mlflow.start_run(description='bench-simplification', log_system_metrics=log_system_metrics)
        for file in files:
            mlflow.log_input(MetaDataset(CodeDatasetSource({}), name=file.name))

    with run_ctx, ZODBTreeBucket() as forest:
        forest.update(load_forest(files), commit=True)

        oid_list = list(forest.oids())

        for size in sizes:
            for nb_worker in workers:
                bench_run_ctx = (
                    mlflow.start_run(
                        description='bench-simplification', nested=True, log_system_metrics=log_system_metrics
                    )
                    if mlflow.active_run()
                    else nullcontext()
                )

                with (
                    bench_run_ctx,
                    console.status(f"[cyan]Benchmarking instances size={size}, workers={nb_worker}..."),
                    ZODBTreeBucket() as bench_forest,
                ):
                    console.print("\x1b[1E")
                    trees = (forest[oid].copy() for oid in oid_list[:size])
                    bench_forest.update(trees, commit=True)
                    mlflow.log_params({'size': size, 'workers': nb_worker})

                    start = time.perf_counter()
                    try:
                        rewrite(
                            bench_forest,
                            tau=tau,
                            decay=decay,
                            epoch=epoch,
                            min_support=min_support,
                            debug=debug,
                            max_workers=nb_worker,
                            commit=min(1024, int(size / nb_worker)),
                        )
                    finally:
                        elapsed = time.perf_counter() - start
                        console.print(f"[magenta]Run completed (size={size}, workers={nb_worker}) in {elapsed:.2f}s[/]")

                gc.collect()

import shutil
import subprocess
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import mlflow
import more_itertools
import typer
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from platformdirs import user_cache_path
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from typer.main import get_command

from architxt.bucket.zodb import ZODBTreeBucket
from architxt.generator import gen_instance
from architxt.inspector import ForestInspector
from architxt.metrics import Metrics
from architxt.schema import Group, Relation, Schema
from architxt.similarity import DECAY
from architxt.simplification.tree_rewriting import rewrite

from .export import app as export_app
from .loader import app as loader_app
from .utils import console, load_forest, show_metrics, show_schema

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

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


@app.command(help="Simplify a bunch of databased together.")
def simplify(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
    epoch: int = typer.Option(100, help="Number of iteration for tree rewriting.", min=1),
    min_support: int = typer.Option(20, help="Minimum support for tree patterns.", min=1),
    workers: int | None = typer.Option(
        None, help="Number of parallel worker processes to use. Defaults to the number of available CPU cores.", min=1
    ),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    metrics: bool = typer.Option(False, help="Show metrics of the simplification."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
    log_system_metrics: bool = typer.Option(False, help="Enable logging of system metrics to MLFlow."),
) -> None:
    run_ctx: AbstractContextManager = nullcontext()

    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        run_ctx = mlflow.start_run(description='simplification', log_system_metrics=log_system_metrics)
        for file in files:
            mlflow.log_input(MetaDataset(CodeDatasetSource({}), name=file.name))

    with run_ctx, ZODBTreeBucket(storage_path=output) as forest:
        forest.update(load_forest(files), commit=True)

        console.print(
            f'[blue]Rewriting {len(forest)} trees with tau={tau}, decay={decay}, epoch={epoch}, min_support={min_support}[/]'
        )
        result_metrics = rewrite(
            forest, tau=tau, decay=decay, epoch=epoch, min_support=min_support, debug=debug, max_workers=workers
        )

        # Generate schema
        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)

        if metrics:
            show_metrics(result_metrics)


@app.command(help="Simplify a bunch of databased together.")
def simplify_llm(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
    min_support: int = typer.Option(20, help="Minimum support for vocab.", min=1),
    vocab_similarity: float = typer.Option(0.6, help="The vocabulary similarity threshold.", min=0, max=1),
    refining_steps: int = typer.Option(0, help="Number of refining steps."),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    intermediate_output: Path | None = typer.Option(None, help="Path to save intermediate results."),
    debug: bool = typer.Option(False, help="Enable debug mode for more verbose output."),
    metrics: bool = typer.Option(False, help="Show metrics of the simplification."),
    log: bool = typer.Option(False, help="Enable logging to MLFlow."),
    log_system_metrics: bool = typer.Option(False, help="Enable logging of system metrics to MLFlow."),
    model_provider: str = typer.Option('huggingface', help="Provider of the model."),
    model: str = typer.Option('HuggingFaceTB/SmolLM2-135M-Instruct', help="Model to use for the LLM."),
    max_tokens: int = typer.Option(2048, help="Maximum number of tokens to generate."),
    local: bool = typer.Option(True, help="Use local model."),
    openvino: bool = typer.Option(False, help="Enable Intel OpenVINO optimizations."),
    rate_limit: float | None = typer.Option(None, help="Rate limit for the LLM."),
    estimate: bool = typer.Option(False, help="Estimate the number of tokens to generate."),
    temperature: float = typer.Option(0.2, help="Temperature for the LLM."),
) -> None:
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        from architxt.simplification.llm import estimate_tokens, llm_rewrite
    except ImportError:
        typer.secho(
            "LLM simplification is unavailable because optional dependencies are missing.\n"
            "Install them with: `pip install architxt[llm]`\n"
            "If using an external provider, also install the appropriate bridge, e.g. `pip install langchain-openai`",
            fg="yellow",
            err=True,
        )
        raise typer.Exit(code=2)

    run_ctx: AbstractContextManager = nullcontext()

    if log:
        console.print(f'[green]MLFlow logging enabled. Logs will be send to {mlflow.get_tracking_uri()}[/]')
        run_ctx = mlflow.start_run(description='llm simplification', log_system_metrics=log_system_metrics)
        mlflow.langchain.autolog()
        mlflow.log_params(
            {
                'model_provider': model_provider,
                'model': model,
                'max_tokens': max_tokens,
                'local': local,
                'openvino': openvino,
                'rate_limit': rate_limit,
                'temperature': temperature,
            }
        )
        for file in files:
            mlflow.log_input(MetaDataset(CodeDatasetSource({}), name=file.name))

    rate_limiter = InMemoryRateLimiter(requests_per_second=rate_limit) if rate_limit else None
    llm: BaseChatModel

    if model_provider == 'huggingface' and local:
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model,
            task='text-generation',
            device_map=None if openvino else 'auto',
            backend='openvino' if openvino else 'pt',
            model_kwargs={'export': True} if openvino else {'torch_dtype': 'auto'},
            pipeline_kwargs={
                'use_cache': True,
                'do_sample': True,
                'return_full_text': False,
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'repetition_penalty': 1.1,
                'num_return_sequences': 1,
                'pad_token_id': 0,
            },
        )
        llm = ChatHuggingFace(llm=pipeline, rate_limiter=rate_limiter)

    else:
        llm = init_chat_model(
            model_provider=model_provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
        )

    if estimate:
        num_input_tokens, num_output_tokens, num_queries = estimate_tokens(
            load_forest(files),
            llm=llm,
            max_tokens=max_tokens,
            refining_steps=refining_steps,
        )
        console.print(f'[blue]Estimated number of tokens: input={num_input_tokens}, output={num_output_tokens}[/]')
        if rate_limit:
            console.print(
                f'[blue]Estimated number of queries: {num_queries} queries (~{num_queries / rate_limit:.2f}s)[/]'
            )
        else:
            console.print(f'[blue]Estimated number of queries: {num_queries} queries[/]')
        return

    with run_ctx, ZODBTreeBucket(storage_path=output) as forest:
        forest.update(load_forest(files), commit=True)

        console.print(f'[blue]Rewriting {len(forest)} trees with model={model}[/]')
        result_metrics = anyio.run(
            llm_rewrite,
            forest,
            llm,
            max_tokens,
            tau,
            decay,
            min_support,
            vocab_similarity,
            refining_steps,
            debug,
            intermediate_output,
        )

        # Generate schema
        schema = Schema.from_forest(forest, keep_unlabelled=False)
        show_schema(schema)

        if metrics:
            show_metrics(result_metrics)


@app.command(help="Display statistics of a dataset.")
def inspect(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
) -> None:
    """Display overall statistics."""
    inspector = ForestInspector()
    forest = load_forest(files)
    forest = inspector(forest)

    # Display the schema
    schema = Schema.from_forest(forest, keep_unlabelled=False)
    show_schema(schema)

    # Display the largest tree
    console.print(Panel(str(inspector.largest_tree), title="Largest Tree"))

    # Entity Count
    tables = []
    for chunk in more_itertools.chunked_even(inspector.entity_count.most_common(), 10):
        entity_table = Table()
        entity_table.add_column("Entity", style="cyan", no_wrap=True)
        entity_table.add_column("Count", style="magenta")

        for entity, count in chunk:
            entity_table.add_row(entity, str(count))

        tables.append(entity_table)

    # Display statistics
    stats_table = Table()
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="magenta")

    stats_table.add_row("Total Trees", str(inspector.total_trees))
    stats_table.add_row("Total Entities", str(inspector.total_entities))
    stats_table.add_row("Average Tree Height", f"{inspector.avg_height:.3f}")
    stats_table.add_row("Maximum Tree Height", str(inspector.max_height))
    stats_table.add_row("Average Tree size", f"{inspector.avg_size:.3f}")
    stats_table.add_row("Maximum Tree size", str(inspector.max_size))
    stats_table.add_row("Average Branching", f"{inspector.avg_branching:.3f}")
    stats_table.add_row("Maximum Branching", str(inspector.max_children))

    console.print(Columns([*tables, stats_table], equal=True))


@app.command(help="Simplify a bunch of databased together.")
def compare(
    file1: Path = typer.Argument(..., exists=True, readable=True, help="Path of the first data file to load."),
    file2: Path = typer.Argument(..., exists=True, readable=True, help="Path of the first data file to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
) -> None:
    # Metrics
    inspector1 = ForestInspector()
    inspector2 = ForestInspector()

    with ZODBTreeBucket(storage_path=file1, read_only=True) as bucket:
        for _ in inspector1(bucket):
            pass
        metrics = Metrics(bucket, tau=tau, decay=decay)

    with ZODBTreeBucket(storage_path=file2, read_only=True) as bucket:
        for _ in inspector2(bucket):
            pass
        metrics.update(bucket)

    show_metrics(metrics)

    # Entity Count
    tables = []
    entities = inspector1.entity_count.keys() | inspector2.entity_count.keys()
    for chunk in more_itertools.chunked_even(entities, 10):
        entity_table = Table()
        entity_table.add_column("Entity", style="cyan", no_wrap=True)
        entity_table.add_column("Count File1", style="magenta")
        entity_table.add_column("Count File2", style="magenta")

        for entity in chunk:
            entity_table.add_row(
                entity,
                str(inspector1.entity_count[entity]),
                str(inspector2.entity_count[entity]),
            )

        tables.append(entity_table)

    # Display statistics
    stats_table = Table()
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value File1", style="magenta")
    stats_table.add_column("Value File2", style="magenta")

    stats_table.add_row("Total Trees", str(inspector1.total_trees), str(inspector2.total_trees))
    stats_table.add_row("Total Entities", str(inspector1.total_entities), str(inspector2.total_entities))
    stats_table.add_row("Average Tree Height", f"{inspector1.avg_height:.3f}", f"{inspector2.avg_height:.3f}")
    stats_table.add_row("Maximum Tree Height", str(inspector1.max_height), str(inspector2.max_height))
    stats_table.add_row("Average Tree size", f"{inspector1.avg_size:.3f}", f"{inspector2.avg_size:.3f}")
    stats_table.add_row("Maximum Tree size", str(inspector1.max_size), str(inspector2.max_size))
    stats_table.add_row("Average Branching", f"{inspector1.avg_branching:.3f}", f"{inspector2.avg_branching:.3f}")
    stats_table.add_row("Maximum Branching", str(inspector1.max_children), str(inspector2.max_children))

    console.print(Columns([*tables, stats_table], equal=True))


@app.command(name='generate', help="Generate synthetic instance.")
def instance_generator(
    *,
    sample: int = typer.Option(100, help="Number of sentences to sample from the corpus.", min=1),
    output: Path | None = typer.Option(None, help="Path to save the result."),
) -> None:
    """Generate synthetic database instances."""
    schema = Schema.from_description(
        groups={
            Group(name='SOSY', entities={'SOSY', 'ANATOMIE', 'SUBSTANCE'}),
            Group(name='TREATMENT', entities={'SUBSTANCE', 'DOSAGE', 'ADMINISTRATION', 'FREQUENCY'}),
            Group(name='EXAM', entities={'DIAGNOSTIC_PROCEDURE', 'ANATOMIE'}),
        },
        relations={
            Relation(name='PRESCRIPTION', left='SOSY', right='TREATMENT'),
            Relation(name='EXAM_RESULT', left='EXAM', right='SOSY'),
        },
    )
    show_schema(schema)

    with (
        ZODBTreeBucket(storage_path=output) as forest,
        console.status("[cyan]Generating synthetic instances..."),
    ):
        trees = gen_instance(schema, size=sample, generate_collections=False)
        forest.update(trees, commit=True)

        console.print(f'[green]Generated {len(forest)} synthetic instances.[/]')


@app.command(name='cache-clear', help='Clear all the cache of ArchiTXT')
def clear_cache(
    *,
    force: bool = typer.Option(False, help="Force the deletion of the cache without asking."),
) -> None:
    cache_path = user_cache_path('architxt')

    if not cache_path.exists():
        console.print("[yellow]Cache is already empty or does not exist. Doing nothing.[/]")
        return

    if not force and not typer.confirm('All the cache data will be deleted. Are you sure?'):
        typer.Abort()

    shutil.rmtree(cache_path)
    console.print("[green]Cache cleared.[/]")


# Click command used for Sphinx documentation
_click_command = get_command(app)

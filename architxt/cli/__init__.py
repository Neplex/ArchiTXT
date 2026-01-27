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
from architxt.metrics import Metrics, redundancy_score
from architxt.schema import Group, Relation, Schema
from architxt.similarity import DECAY
from architxt.simplification.tree_rewriting import rewrite

from .export import app as export_app
from .loader import app as loader_app
from .utils import console, get_schema_metrics, load_forest, show_metrics, show_schema, show_valid_trees_metrics

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


@app.command(help="Cleanup a forest retaining only the valid tree structure")
def cleanup(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
    output: Path | None = typer.Option(None, help="Path to save the result."),
    metrics: bool = typer.Option(False, help="Show metrics of the simplification."),
) -> None:
    with (
        ZODBTreeBucket() as tmp_forest,
        ZODBTreeBucket(storage_path=output) as output_forest,
    ):
        tmp_forest.update(load_forest(files), commit=True)
        schema = Schema.from_forest(tmp_forest, keep_unlabelled=False)

        show_schema(schema)

        if metrics:
            result_metrics = Metrics(tmp_forest, tau=tau, decay=decay)

        trees = schema.extract_valid_trees(tmp_forest)
        output_forest.update(trees, commit=True)

        if metrics:
            result_metrics.update(output_forest)
            show_metrics(result_metrics)


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
            show_valid_trees_metrics(result_metrics, schema, forest, epoch + 1, log)


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
            show_valid_trees_metrics(result_metrics, schema, forest, refining_steps + 1, log)


@app.command(help="Display statistics of a dataset.")
def inspect(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Path of the data files to load."),
    redundancy: bool = typer.Option(False, help="Compute redundancy metrics."),
) -> None:
    """Display overall statistics."""
    inspector = ForestInspector()

    with ZODBTreeBucket() as forest:
        trees = inspector(load_forest(files))
        forest.update(trees, commit=True)
        schema = Schema.from_forest(inspector(forest), keep_unlabelled=False)

        # Display the schema
        show_schema(schema)

        # Display the largest tree
        console.print(Panel(str(inspector.largest_tree), title="Largest Tree"))

        # Entity Count
        tables = []
        for chunk in more_itertools.chunked_even(inspector.entity_count.most_common(), 10):
            entity_table = Table(title='Entity Counts')
            entity_table.add_column("Entity", style="cyan", no_wrap=True)
            entity_table.add_column("Count", style="magenta")

            for entity, count in chunk:
                entity_table.add_row(entity, str(count))

            tables.append(entity_table)

        # Display statistics
        stats_table = Table(title='Statistics')
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="magenta")

        stats_table.add_row("Total Trees", str(inspector.total_trees))
        stats_table.add_row("Total Entities", str(inspector.total_entities))
        stats_table.add_row("Total Groups", str(inspector.total_groups))
        stats_table.add_row("Total Relations", str(inspector.total_relations))
        stats_table.add_row("Average Tree Height", f"{inspector.avg_height:.3f}")
        stats_table.add_row("Maximum Tree Height", str(inspector.max_height))
        stats_table.add_row("Average Tree size", f"{inspector.avg_size:.3f}")
        stats_table.add_row("Maximum Tree size", str(inspector.max_size))
        stats_table.add_row("Average Branching", f"{inspector.avg_branching:.3f}")
        stats_table.add_row("Maximum Branching", str(inspector.max_children))

        if redundancy:
            datasets = schema.extract_datasets(forest)
            for tau in (1.0, 0.7, 0.5):
                redundancy = sum(redundancy_score(ds, tau=tau) for ds in datasets.values()) / len(datasets)
                stats_table.add_row(f"Redundant Trees ({tau}:.1f)", f"{redundancy:.3f}")

        console.print(Columns([*tables, stats_table, get_schema_metrics(schema)], equal=True))


@app.command(help="Simplify a bunch of databased together.")
def compare(
    src: Path = typer.Argument(..., exists=True, readable=True, help="Path of the data file to compare to."),
    dst: Path = typer.Argument(..., exists=True, readable=True, help="Path of the data file to compare."),
    *,
    tau: float = typer.Option(0.7, help="The similarity threshold.", min=0, max=1),
    decay: float = typer.Option(DECAY, help="The similarity decay factor.", min=0.001),
) -> None:
    # Metrics
    inspector_src = ForestInspector()
    inspector_dst = ForestInspector()

    with (
        ZODBTreeBucket() as forest_src,
        ZODBTreeBucket() as forest_dst,
    ):
        trees_src = inspector_src(load_forest([src]))
        forest_src.update(trees_src, commit=True)
        metrics = Metrics(forest_src, tau=tau, decay=decay)

        trees_dst = inspector_dst(load_forest([dst]))
        forest_dst.update(trees_dst, commit=True)
        metrics.update(forest_dst)
        schema = Schema.from_forest(forest_dst, keep_unlabelled=False)

        show_metrics(metrics)
        show_valid_trees_metrics(metrics, schema, forest_dst, 0, False)

    # Entity Count
    tables = []
    entities = inspector_src.entity_count.keys() | inspector_dst.entity_count.keys()
    for chunk in more_itertools.chunked_even(entities, 10):
        entity_table = Table()
        entity_table.add_column("Entity", style="cyan", no_wrap=True)
        entity_table.add_column("Count source", style="magenta")
        entity_table.add_column("Count destination", style="magenta")

        for entity in chunk:
            entity_table.add_row(
                entity,
                str(inspector_src.entity_count[entity]),
                str(inspector_dst.entity_count[entity]),
            )

        tables.append(entity_table)

    # Display statistics
    stats_table = Table()
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value source", style="magenta")
    stats_table.add_column("Value destination", style="magenta")

    stats_table.add_row("Total Trees", str(inspector_src.total_trees), str(inspector_dst.total_trees))
    stats_table.add_row("Total Entities", str(inspector_src.total_entities), str(inspector_dst.total_entities))
    stats_table.add_row("Total Entities", str(inspector_src.total_entities), str(inspector_dst.total_entities))
    stats_table.add_row("Total Groups", str(inspector_src.total_groups), str(inspector_dst.total_groups))
    stats_table.add_row("Average Tree Height", f"{inspector_src.avg_height:.3f}", f"{inspector_dst.avg_height:.3f}")
    stats_table.add_row("Maximum Tree Height", str(inspector_src.max_height), str(inspector_dst.max_height))
    stats_table.add_row("Average Tree size", f"{inspector_src.avg_size:.3f}", f"{inspector_dst.avg_size:.3f}")
    stats_table.add_row("Maximum Tree size", str(inspector_src.max_size), str(inspector_dst.max_size))
    stats_table.add_row("Average Branching", f"{inspector_src.avg_branching:.3f}", f"{inspector_dst.avg_branching:.3f}")
    stats_table.add_row("Maximum Branching", str(inspector_src.max_children), str(inspector_dst.max_children))

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

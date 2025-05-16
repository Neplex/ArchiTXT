import asyncio
import hashlib
import tarfile
import zipfile
from collections.abc import AsyncGenerator, Iterable, Sequence
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, BinaryIO

import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from platformdirs import user_cache_path
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from architxt.bucket.zodb import ZODBTreeBucket
from architxt.nlp.brat import load_brat_dataset
from architxt.nlp.entity_resolver import EntityResolver, ScispacyResolver
from architxt.nlp.parser import Parser
from architxt.tree import Tree
from architxt.utils import BATCH_SIZE

if TYPE_CHECKING:
    from architxt.nlp.model import AnnotatedSentence

__all__ = ['raw_load_corpus']

console = Console()

CACHE_DIR = user_cache_path('architxt')


async def _get_cache_key(
    archive_file: BytesIO | BinaryIO,
    *,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    language: str,
    resolver: EntityResolver | None = None,
) -> str:
    """Generate a cache key based on the archive file's content and settings."""
    cursor = archive_file.tell()
    file_hash = await asyncio.to_thread(hashlib.file_digest, archive_file, hashlib.md5)
    archive_file.seek(cursor)

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

    return file_hash.hexdigest()


def open_archive(archive_file: BytesIO | BinaryIO) -> zipfile.ZipFile | tarfile.TarFile:
    cursor = archive_file.tell()
    signature = archive_file.read(4)
    archive_file.seek(cursor)

    if signature.startswith(b'PK\x03\x04'):  # ZIP file signature
        return zipfile.ZipFile(archive_file)

    if signature.startswith(b'\x1f\x8b'):  # GZIP signature (tar.gz)
        return tarfile.TarFile.open(fileobj=archive_file)

    msg = "Unsupported file format"
    raise ValueError(msg)


async def _load_or_cache_corpus(  # noqa: C901
    archive_file: str | Path | BytesIO | BinaryIO,
    queue: asyncio.Queue[Tree],
    progress: Progress,
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
    sample: int | None = None,
) -> None:
    """
    Load the corpus from the disk or cache.

    :param archive_file: A path or an in-memory file object of the corpus archive.
    :param entities_filter: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param entities_mapping: A dictionary mapping entities names to new values. If None, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :param parser: The NLP parser to use.
    :param language: The language to use for parsing.
    :param name: The corpus name.
    :param resolver: An optional entity resolver to use.
    :param cache: Whether to cache the computed forest or not.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    should_close = False
    corpus_cache_path: Path | None = None

    if isinstance(archive_file, str | Path):
        archive_file = Path(archive_file).open('rb')  # noqa: SIM115
        should_close = True

    try:
        key = await _get_cache_key(
            archive_file,
            entities_filter=entities_filter,
            entities_mapping=entities_mapping,
            relations_filter=relations_filter,
            relations_mapping=relations_mapping,
            language=language,
            resolver=resolver,
        )
        if cache:
            directory = CACHE_DIR / 'corpus_cache'
            directory.mkdir(parents=True, exist_ok=True)
            corpus_cache_path = directory / key

        if mlflow.active_run():
            mlflow.log_input(
                MetaDataset(
                    CodeDatasetSource(
                        {
                            'entities_filter': sorted(entities_filter or []),
                            'relations_filter': sorted(relations_filter or []),
                            'entities_mapping': entities_mapping,
                            'relations_mapping': relations_mapping,
                            'cache_file': str(corpus_cache_path.absolute()) if corpus_cache_path else None,
                        }
                    ),
                    name=name or archive_file.name,
                    digest=key,
                )
            )

        with ZODBTreeBucket(storage_path=corpus_cache_path) as forest:
            count = 0

            if cache and len(forest):  # Attempt to load from cache if available
                for tree in progress.track(
                    forest,
                    description=f'[green]Loading corpus {archive_file.name} from cache...[/]',
                    total=sample,
                ):
                    await queue.put(tree.copy())
                    count += 1

                    if sample and count >= sample:
                        break

            else:  # Load data from disk
                with (
                    open_archive(archive_file) as corpus,
                    TemporaryDirectory() as tmp_dir,
                ):
                    # Extract archive contents to a temporary directory
                    await asyncio.to_thread(corpus.extractall, path=tmp_dir)

                    # Parse sentences and enrich the forest
                    sentences: Iterable[AnnotatedSentence] = load_brat_dataset(
                        Path(tmp_dir),
                        entities_filter=entities_filter,
                        relations_filter=relations_filter,
                        entities_mapping=entities_mapping,
                        relations_mapping=relations_mapping,
                    )

                    sentences = progress.track(
                        sentences,
                        description=f'[yellow]Loading corpus {archive_file.name} from disk...[/]',
                    )

                    batch = []
                    async for _, tree in parser.parse_batch(sentences, language=language, resolver=resolver):
                        batch.append(tree)

                        if not sample or count < sample:
                            await queue.put(tree.copy())
                            count += 1

                        if len(batch) >= BATCH_SIZE:
                            await asyncio.to_thread(forest.update, batch)
                            batch.clear()

                    if batch:
                        await asyncio.to_thread(forest.update, batch)

    except Exception as e:
        console.print(f'[red]Error while processing corpus:[/] {e}')
        raise

    finally:
        if should_close:
            archive_file.close()


async def raw_load_corpus(
    corpus_archives: Sequence[str | Path | BytesIO | BinaryIO],
    languages: Sequence[str],
    *,
    parser: Parser,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    resolver_name: str | None = None,
    cache: bool = True,
    sample: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> AsyncGenerator[Tree, None]:
    """
    Asynchronously loads a set of corpus from disk or in-memory archives, parses it, and returns the enriched forest.

    This function handles both local and in-memory corpus archives, processes the data based on the specified filters
    and mappings, and uses the provided CoreNLP server for parsing.
    Optionally, caching can be enabled to avoid repeated computations.
    The resulting forest is not a valid database instance it needs to be passed to the automatic structuration algorithm first.

    :param corpus_archives: A list of corpus archive sources, which can be:
        - Paths to files on disk, or
        - In-memory file-like objects.
        The list can include both local and in-memory sources, and its size should match the length of `languages`.
    :param languages: A list of languages corresponding to each corpus archive. The number of languages must match the number of archives.
    :param parser: The parser to use to parse the sentences.
    :param entities_filter: A set of entity types to exclude from the output. If py:`None`, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If py:`None`, no filtering is applied.
    :param entities_mapping: A dictionary mapping entities names to new values. If py:`None`, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If py:`None`, no mapping is applied.
    :param resolver_name: The name of the entity resolver to use. If py:`None`, no entity resolution is performed.
    :param cache: A boolean flag indicating whether to cache the computed forest for faster future access.
    :param sample: The number of examples to take in each corpus.
    :param batch_size: The number of sentences to process in each batch.
        This parameter is used to control the memory usage.

    :returns: A forest containing the parsed and enriched trees.
    """
    with (
        parser as parser_ctx,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        resolver_ctx = (
            ScispacyResolver(cleanup=True, translate=True, kb_name=resolver_name) if resolver_name else nullcontext()
        )

        async with resolver_ctx as resolver:
            queue: asyncio.Queue[Tree] = asyncio.Queue(batch_size)
            pending = {
                asyncio.create_task(
                    _load_or_cache_corpus(
                        corpus,
                        queue,
                        progress,
                        entities_filter=entities_filter,
                        relations_filter=relations_filter,
                        entities_mapping=entities_mapping,
                        relations_mapping=relations_mapping,
                        parser=parser_ctx,
                        language=language,
                        resolver=resolver,
                        cache=cache,
                        sample=sample,
                    )
                )
                for corpus, language in zip(corpus_archives, languages, strict=True)
            }

            try:
                while pending or not queue.empty():
                    while not queue.empty():
                        yield await queue.get()

                    if pending:
                        _, pending = await asyncio.wait(pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)

            finally:
                for task in pending:
                    task.cancel()

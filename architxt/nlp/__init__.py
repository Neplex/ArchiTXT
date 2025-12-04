from __future__ import annotations

import hashlib
import tarfile
import zipfile
from contextlib import ExitStack, nullcontext
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, BinaryIO

import anyio
import anyio.to_thread
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from platformdirs import user_cache_path
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from architxt.bucket.zodb import ZODBTreeBucket
from architxt.nlp.brat import load_brat_dataset
from architxt.utils import BATCH_SIZE

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Sequence
    from io import BytesIO

    from anyio.abc import ObjectSendStream
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from architxt.nlp.entity_extractor import EntityExtractor
    from architxt.nlp.entity_resolver import EntityResolver
    from architxt.nlp.model import AnnotatedSentence
    from architxt.nlp.parser import Parser
    from architxt.tree import Tree

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
    extractor: EntityExtractor | None = None,
) -> str:
    """Generate a cache key based on the archive file's content and settings."""
    cursor = archive_file.tell()
    file_hash = await anyio.to_thread.run_sync(hashlib.file_digest, archive_file, hashlib.md5)
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
    if extractor:
        file_hash.update(extractor.name.encode())

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
    send_stream: ObjectSendStream[Tree],
    progress: Progress,
    parser: Parser,
    language: str,
    entities_filter: set[str] | None = None,
    relations_filter: set[str] | None = None,
    entities_mapping: dict[str, str] | None = None,
    relations_mapping: dict[str, str] | None = None,
    resolver: EntityResolver | None = None,
    extractor: EntityExtractor | None = None,
    cache: bool = True,
    sample: int | None = None,
    name: str | None = None,
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
    :param extractor: An optional entity extractor to use.
    :param cache: Whether to cache the computed forest or not.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    should_close = False
    corpus_cache_path: Path | None = None

    if isinstance(archive_file, str | Path):
        archive_file = Path(archive_file).open('rb')  # noqa: ASYNC230, SIM115
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
            extractor=extractor,
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

        # If caching enabled and cache exists, attempt to load from cache if available
        if cache and corpus_cache_path and corpus_cache_path.exists():
            with ZODBTreeBucket(storage_path=corpus_cache_path, read_only=True) as forest:
                if len(forest):
                    for tree in progress.track(
                        islice(forest, sample) if sample else forest,
                        description=f'[green]Loading corpus {archive_file.name} from cache...[/]',
                        total=sample,
                    ):
                        await send_stream.send(tree.copy())
                    return

        # No cache or cache disabled: extract and parse
        with (
            open_archive(archive_file) as corpus,
            TemporaryDirectory() as tmp_dir,
        ):
            # Extract archive contents to a temporary directory
            await anyio.to_thread.run_sync(corpus.extractall, tmp_dir)

            # Parse sentences and enrich the forest
            sentences: Iterable[AnnotatedSentence] | AsyncIterable[AnnotatedSentence] = progress.track(
                load_brat_dataset(
                    Path(tmp_dir),
                    entities_filter=entities_filter,
                    relations_filter=relations_filter,
                    entities_mapping=entities_mapping,
                    relations_mapping=relations_mapping,
                ),
                description=f'[yellow]Loading corpus {archive_file.name} from disk...[/]',
            )

            # Extract more entities
            if extractor:
                sentences = extractor.enrich(sentences)

            # Resolve entities
            if resolver:
                sentences = resolver.batch_sentences(sentences)

            # If cache disabled: sample-only short-circuit
            if not cache:
                count = 0
                async for _, tree in parser.parse_batch(sentences, language=language):
                    if sample and count >= sample:
                        break

                    await send_stream.send(tree.copy())
                    count += 1

            else:
                with ZODBTreeBucket(storage_path=corpus_cache_path) as forest, ExitStack() as transaction_stack:
                    transaction_stack.enter_context(forest.transaction())
                    count = 0

                    async for _, tree in parser.parse_batch(sentences, language=language):
                        forest.add(tree)
                        count += 1

                        if count % BATCH_SIZE == 0:  # Commit the current transaction and create a new one
                            transaction_stack.close()
                            transaction_stack.enter_context(forest.transaction())

                        if not sample or count <= sample:
                            await send_stream.send(tree.copy())

    except Exception as e:
        console.print(f'[red]Error while processing corpus:[/] {e}')
        raise

    finally:
        await send_stream.aclose()
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
    resolver: EntityResolver | None = None,
    extractor: EntityExtractor | None = None,
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
    :param extractor: The entity extractor to use. If py:`None`, no extra entity extraction is performed.
    :param resolver: The entity resolver to use. If py:`None`, no entity resolution is performed.
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
        async with resolver or nullcontext() as resolver, anyio.create_task_group() as tg:
            send_stream: MemoryObjectSendStream[Tree]
            receive_stream: MemoryObjectReceiveStream[Tree]
            send_stream, receive_stream = anyio.create_memory_object_stream(batch_size)

            for corpus, language in zip(corpus_archives, languages, strict=True):
                tg.start_soon(
                    _load_or_cache_corpus,
                    corpus,
                    send_stream.clone(),
                    progress,
                    parser_ctx,
                    language,
                    entities_filter,
                    relations_filter,
                    entities_mapping,
                    relations_mapping,
                    resolver,
                    extractor,
                    cache,
                    sample,
                )

            send_stream.close()
            async with receive_stream:
                async for tree in receive_stream:
                    yield tree

import asyncio
import hashlib
import random
import tarfile
import zipfile
from collections.abc import Sequence
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from rich.console import Console
from tqdm.auto import tqdm

from architxt.nlp.brat import load_brat_dataset
from architxt.nlp.entity_resolver import EntityResolver, ScispacyResolver
from architxt.nlp.parser import Parser
from architxt.tree import Forest, Tree
from architxt.utils import read_cache, write_cache

__all__ = ['raw_load_corpus']

console = Console()


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


async def _load_or_cache_corpus(
    archive_file: str | Path | BytesIO | BinaryIO,
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
) -> Forest:
    """
    Load the corpus from disk or cache.

    :param archive_file: A path or an in-memory file object of the corpus archive.
    :param entities_filter: A set of entity types to exclude from the output. If None, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If None, no filtering is applied.
    :param entities_mapping: A dictionary mapping entity names to new values. If None, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If None, no mapping is applied.
    :param parser: The NLP parser to use.
    :param language: The language to use for parsing.
    :param name: The corpus name.
    :param resolver: An optional entity resolver to use.
    :param cache: Whether to cache the computed forest or not.
    :param sample: The number of examples to get from the corpus.

    :returns: A list of parsed trees representing the enriched corpus.
    """
    should_close = False

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
        corpus_cache_path = Path(f'{key}.pkl')

        if mlflow.active_run():
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
                    name=name or archive_file.name,
                    digest=key,
                )
            )

        # Attempt to load from cache if available
        if cache and corpus_cache_path.exists():
            console.print(f'[green]Loading corpus from cache:[/] {corpus_cache_path.absolute()}')
            return await read_cache(corpus_cache_path)

        console.print(f'[yellow]Loading corpus from disk:[/] {archive_file.name}')
        corpus: zipfile.ZipFile | tarfile.TarFile

        # If the cache does not exist, process the archive
        with (
            open_archive(archive_file) as corpus,
            TemporaryDirectory() as tmp_dir,
        ):
            # Extract archive contents to a temporary directory
            await asyncio.to_thread(corpus.extractall, path=tmp_dir)
            tmp_path = Path(tmp_dir)

            # Parse sentences and enrich the forest
            sentences = load_brat_dataset(
                tmp_path,
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping=entities_mapping,
                relations_mapping=relations_mapping,
            )

            if sample:
                sentences = tqdm(random.sample(tuple(sentences), sample))

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
) -> list[Tree]:
    """
    Asynchronously loads a set of corpus from disk or in-memory archives, parses it, and returns the enriched forest.

    This function handles both local and in-memory corpus archives, processes the data based on the specified filters
    and mappings, and uses the provided CoreNLP server for parsing.
    Optionally, caching can be enabled to avoid repeated computations.
    The resulting forest is not a valid database instance it need to be passed to the automatic structuration algorithm first.

    :param corpus_archives: A list of corpus archive sources, which can be:
        - Paths to files on disk, or
        - In-memory file-like objects.
        The list can include both local and in-memory sources, and its size should match the length of `languages`.
    :param languages: A list of languages corresponding to each corpus archive. The number of languages must match the number of archives.
    :param parser: The parser to use to parse the sentences.
    :param entities_filter: A set of entity types to exclude from the output. If py:`None`, no filtering is applied.
    :param relations_filter: A set of relation types to exclude from the output. If py:`None`, no filtering is applied.
    :param entities_mapping: A dictionary mapping entity names to new values. If py:`None`, no mapping is applied.
    :param relations_mapping: A dictionary mapping relation names to new values. If py:`None`, no mapping is applied.
    :param resolver_name: The name of the entity resolver to use. If py:`None`, no entity resolution is performed.
    :param cache: A boolean flag indicating whether to cache the computed forest for faster future access.
    :param sample: The number of examples to take in each corpus.

    :returns: A forest containing the parsed and enriched trees.
    """
    with parser as paser_ctx:
        resolver_ctx = (
            ScispacyResolver(cleanup=True, translate=True, kb_name=resolver_name) if resolver_name else nullcontext()
        )

        async with resolver_ctx as resolver:
            forests = await asyncio.gather(
                *[
                    _load_or_cache_corpus(
                        corpus,
                        entities_filter=entities_filter,
                        relations_filter=relations_filter,
                        entities_mapping=entities_mapping,
                        relations_mapping=relations_mapping,
                        parser=paser_ctx,
                        language=language,
                        resolver=resolver,
                        cache=cache,
                        sample=sample,
                    )
                    for corpus, language in zip(corpus_archives, languages, strict=True)
                ]
            )

            return [tree for forest in forests for tree in forest]

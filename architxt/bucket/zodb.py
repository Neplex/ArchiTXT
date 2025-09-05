import gc
import tempfile
import uuid
from collections.abc import (
    AsyncIterable,
    Generator,
    Iterable,
    Iterator,
)
from pathlib import Path
from typing import TYPE_CHECKING, overload

import anyio.to_process
import more_itertools
import transaction
import ZODB.config
from aiostream import stream
from BTrees.OOBTree import OOBTree
from ZODB.Connection import Connection

from architxt.tree import Tree, TreeOID
from architxt.utils import BATCH_SIZE, is_memory_low

from . import TreeBucket

if TYPE_CHECKING:
    from aiostream.core import Stream

__all__ = ['ZODBTreeBucket']


class ZODBTreeBucket(TreeBucket):
    """
    A persistent, scalable container for :py:class:`~architxt.tree.Tree` objects backed by ZODB and RelStorage using SQLite.

    This container uses `ZODB <https://zodb.org/en/latest/>`_'s :py:class:`~BTrees.OOBTree.OOBTree` internally
    with Tree OIDs (UUIDs) as keys. The OIDs are stored as raw bytes to optimize storage space.
    This also enables fast key comparisons as UUID objects do not need to be created during lookups.

    .. note::
        UUIDs are stored as bytes rather than integers, because ZODB only supports integers up to
        64 bits, while UUIDs require 128 bits.

    Without a specified storage path, the container creates a temporary database automatically deleted upon closing.

    >>> bucket = ZODBTreeBucket()
    >>> tree = Tree.fromstring('(S (NP Alice) (VP (VB like) (NNS apples)))')
    >>> bucket.add(tree)
    >>> len(bucket)
    1
    >>> tree.label = 'ROOT'
    >>> transaction.commit()  # Persist changes made to the tree
    >>> tree.label
    'ROOT'
    >>> tree.label = 'S'
    >>> transaction.abort()  # Cancel changes made to the tree
    >>> tree.label
    'ROOT'
    >>> bucket.discard(tree)
    >>> len(bucket)
    0
    >>> bucket.close()
    """

    _db: ZODB.DB
    _connection: Connection
    _data: OOBTree
    _temp_dir: tempfile.TemporaryDirectory | None

    def __init__(
        self,
        storage_path: Path | None = None,
        bucket_name: str = 'architxt',
        read_only: bool = False,
    ) -> None:
        """
        Initialize the bucket and connect to the underlying ZODB storage.

        :param storage_path: Path to the storage directory.
            If None, a temporary location is used to store the database.
        :param bucket_name: Name of the root key under which the internal OOBTree is stored.
        :param read_only: Whether to open the database in read-only mode.
        """
        if storage_path is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix='architxt')
            self._storage_path = Path(self._temp_dir.name)

        else:
            self._temp_dir = None
            self._storage_path = storage_path

        self._bucket_name = bucket_name
        self._read_only = read_only
        self._db = ZODB.config.databaseFromString(f"""
            %import relstorage

            <zodb main>
                <relstorage>
                    keep-history false
                    pack-gc true
                    read-only {'true' if self._read_only else 'false'}
                    <sqlite3>
                        data-dir {self._storage_path}
                        <pragmas>
                            synchronous off
                            foreign_keys off
                            defer_foreign_keys on
                            temp_store memory
                        </pragmas>
                    </sqlite3>
                </relstorage>
            </zodb>
        """)
        self._connection = self._db.open()
        root = self._connection.root()

        if self._bucket_name not in root:
            root[self._bucket_name] = OOBTree()
            transaction.commit()

        self._data = root[self._bucket_name]

    def __reduce__(self) -> tuple[type, tuple[Path, str, bool]]:
        return self.__class__, (self._storage_path, self._bucket_name, self._read_only)

    def close(self) -> None:
        """
        Close the database connection and release associated resources.

        This will:

        - Abort any uncommitted transaction.
        - Close the active database connection.
        - Clean up temporary storage if one was created.
        """
        self._connection.transaction_manager.abort()
        self._connection.close()
        self._db.close()

        if self._temp_dir is not None:  # If a temporary directory was used, clean it up
            self._temp_dir.cleanup()

    def transaction(self) -> transaction.TransactionManager:
        return self._connection.transaction_manager

    def commit(self) -> None:
        self._connection.transaction_manager.commit()

    def update(self, trees: Iterable[Tree], batch_size: int = BATCH_SIZE, _memory_threshold_mb: int = 3_000) -> None:
        """
        Add multiple :py:class:`~architxt.tree.Tree` to the bucket, managing memory via chunked transactions.

        Trees are added in batches to reduce memory footprint.
        When available system memory falls below the threshold,
        the connection cache is minimized and garbage collection is triggered.

        .. warning::
            Only the last chunk is rolled back on error.
            Prior chunks remain committed, potentially leaving the database in a partially updated state.

        :param trees: Trees to add to the bucket.
        :param batch_size: The number of trees to be added at once.
        :param _memory_threshold_mb: Memory threshold (in MB) below which garbage collection is triggered.
        """
        for chunk in more_itertools.chunked(trees, batch_size):
            with self.transaction():
                self._data.update({tree.oid.bytes: tree for tree in chunk})

            if is_memory_low(_memory_threshold_mb):
                self._connection.cacheMinimize()
                gc.collect()

    async def async_update(
        self,
        trees: Iterable[Tree] | AsyncIterable[Tree],
        batch_size: int = BATCH_SIZE,
        _memory_threshold_mb: int = 3_000,
    ) -> None:
        """
        Asynchronously add multiple :py:class:`~architxt.tree.Tree` to the bucket.

        This method mirrors the behavior of :py:meth:`~ZODBTreeBucket.update` but supports asynchronous iteration.
        Internally, it delegates each chunk to a background thread.

        :param trees: Trees to add to the bucket.
        :param batch_size: The number of trees to be added at once.
        :param _memory_threshold_mb: Memory threshold (in MB) below which garbage collection is triggered.
        """
        chunk_stream: Stream[list[Tree]] = stream.chunks(stream.iterate(trees), batch_size)
        chunk: list[Tree]

        async with chunk_stream.stream() as streamer:
            async for chunk in streamer:
                await anyio.to_process.run_sync(self.update, chunk, batch_size, _memory_threshold_mb)

    def add(self, tree: Tree) -> None:
        """Add a single :py:class:`~architxt.tree.Tree` to the bucket."""
        with self.transaction():
            self._data[tree.oid.bytes] = tree

    def discard(self, tree: Tree) -> None:
        """Remove a :py:class:`~architxt.tree.Tree` from the bucket if it exists."""
        with self.transaction():
            self._data.pop(tree.oid.bytes)

    def clear(self) -> None:
        """Remove all :py:class:`~architxt.tree.Tree` objects from the bucket."""
        with self.transaction():
            self._data.clear()

    def oids(self) -> Generator[TreeOID, None, None]:
        for key in self._data:
            yield uuid.UUID(bytes=key)

    @overload
    def __getitem__(self, key: TreeOID) -> Tree: ...

    @overload
    def __getitem__(self, key: Iterable[TreeOID]) -> Iterable[Tree]: ...

    def __getitem__(self, key: TreeOID | Iterable[TreeOID]) -> Tree | Iterable[Tree]:
        if isinstance(key, uuid.UUID):
            return self._data[key.bytes]

        return (self._data[oid.bytes] for oid in key)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Tree):
            return item.oid.bytes in self._data

        if isinstance(item, uuid.UUID):
            return item.bytes in self._data

        return False

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Tree]:
        return self._data.itervalues()

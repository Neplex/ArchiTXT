from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, overload

import transaction
import ZODB.config
from BTrees.OOBTree import OOBTree
from transaction.interfaces import AlreadyInTransaction, NoTransaction
from ZODB.Connection import resetCaches
from zodburi import resolve_uri

from architxt.tree import Tree, TreeOID
from architxt.utils import update_url_queries

from . import TreeBucket

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    from ZODB.Connection import Connection

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

    If no storage is specified, the bucket use a temporary database that is automatically deleted upon closing.

    The bucket is serializable so it can be passed to a subprocess. However, when a temporary database is used,
    the original bucket remains responsible for cleanup. This means the original bucket must stay open for the
    subprocess to access the database safely.

    >>> from architxt.bucket.zodb import ZODBTreeBucket
    >>> from architxt.tree import Tree

    >>> bucket = ZODBTreeBucket()
    >>> tree = Tree.fromstring('(S (NP Alice) (VP (VB like) (NNS apples)))')
    >>> tree.label
    'S'

    >>> with bucket.transaction():
    ...     bucket.add(tree) # Add the tree to the bucket
    >>> len(bucket)
    1
    >>> tree in bucket
    True

    >>> with bucket.transaction():
    ...     tree.label = 'ROOT' # Modify the tree within a transaction
    >>> tree.label
    'ROOT'
    >>> bucket[tree.oid].label
    'ROOT'

    >>> with bucket.transaction():
    ...     tree.label = 'S'
    ...     raise ValueError("rollback")  # Transaction are rolled back on exception
    Traceback (most recent call last):
        ...
    ValueError: rollback

    >>> tree.label
    'ROOT'
    >>> with bucket.transaction():
    ...     bucket.discard(tree)
    >>> len(bucket)
    0
    >>> tree in bucket
    False

    >>> bucket.close()
    """

    _db: ZODB.DB
    _connection: Connection
    _cleanup: bool = False

    def __init__(
        self,
        storage_path: Path | None = None,
        uri: str | None = None,
        bucket_name: str = 'architxt',
        read_only: bool = False,
    ) -> None:
        """
        Initialize the bucket and connect to the underlying ZODB storage.

        Either a `uri` or a `storage_path` may be provided. If both are given, the `uri` takes precedence.
        If neither is provided, a temporary local database is created.

        Local databases (temporary or using `storage_path`) use a SQLite backend to support
        multiprocess concurrency. File-based backends are technically possible but not recommended,
        as they can break parallel execution.

        For large datasets or in a multi host setup, prefer a relational backend via a ZODB URI.
        For example, to use PostgreSQL, specify a URI such as: ``postgresql://user:password@localhost/dbname``.

        :param storage_path: Path to the storage directory for local storage.
        :param uri: ZODB URI string defining the storage backend.
        :param bucket_name: Name of the root key under which the internal OOBTree is stored.
        :param read_only: Whether to open the database in read-only mode.
        """
        self._storage_path = storage_path
        self._uri = uri
        self._bucket_name = bucket_name
        self._read_only = read_only

        self._db = self._get_db()
        self._connection = self._get_connection()

        # Check bucket exist or create it
        root = self._connection.root()
        if self._bucket_name not in root:
            if self._read_only:
                msg = f"Bucket '{self._bucket_name}' does not exist."
                raise KeyError(msg)

            with self.transaction():
                root[self._bucket_name] = OOBTree()

        # Add support for fork on supported systems
        # When the process is fork, the child process should recreate the connection
        if hasattr(os, "register_at_fork"):
            weak_self = weakref.ref(self)
            os.register_at_fork(after_in_child=lambda: (bucket := weak_self()) and bucket._after_fork())

    def _get_db(self) -> ZODB.DB:
        """
        Create and configure the ZODB database.

        URI-based storage takes precedence over local storage path.
        For local storage, if no path is provided, a temporary directory is created.
        We use RelStorage with SQLite as the backend for local storage to allow multi-process access.

        :return: A configured ZODB database instance.
        """
        if self._uri:
            uri = update_url_queries(self._uri, read_only='true') if self._read_only else self._uri
            storage_factory, db_options = resolve_uri(uri)
            storage = storage_factory()
            return ZODB.DB(storage, **db_options)

        if self._storage_path is None:
            if self._read_only:
                msg = "Cannot open a read-only bucket with no storage path specified."
                raise ValueError(msg)

            self._storage_path = Path(tempfile.mkdtemp(prefix='architxt'))
            self._cleanup = True

        return ZODB.config.databaseFromString(f"""
            %import relstorage

            <zodb main>
                <relstorage>
                    keep-history false
                    pack-gc true
                    read-only {'true' if self._read_only else 'false'}
                    <sqlite3>
                        data-dir {self._storage_path}
                        <pragmas>
                            synchronous normal
                            foreign_keys off
                            defer_foreign_keys on
                            temp_store memory
                            journal_mode wal
                            busy_timeout 10000
                        </pragmas>
                    </sqlite3>
                </relstorage>
            </zodb>
        """)

    def _get_connection(self) -> Connection:
        transaction_manager = transaction.TransactionManager(explicit=True)
        return self._db.open(transaction_manager=transaction_manager)

    @property
    def _data(self) -> OOBTree:
        return self._connection.root()[self._bucket_name]

    def __reduce__(self) -> tuple[type, tuple[Path | None, str | None, str, bool]]:
        return self.__class__, (self._storage_path, self._uri, self._bucket_name, self._read_only)

    def _savepoint(self) -> None:
        self._connection.savepoint()

    def _after_fork(self) -> None:
        # We disable database cleanup as it is the responsibility of the parent
        self._cleanup = False
        # We also recreate the connection as it inherit the one from the parent process
        self.sync()

    def close(self) -> None:
        """
        Close the database connection and release associated resources.

        This will:
        - Abort any uncommitted transaction.
        - Close the active database connection.
        - Clean up temporary storage if one was created.
        """
        with contextlib.suppress(NoTransaction):
            self._connection.abort(None)

        self._connection.close()
        self._db.close()

        if self._cleanup:  # If a temporary directory was used, clean it up
            shutil.rmtree(self._storage_path)

    @contextlib.contextmanager
    def transaction(self) -> Generator[None, None, None]:
        try:
            with self._connection.transaction_manager:
                yield

        except AlreadyInTransaction:
            yield

    def sync(self) -> None:
        """
        Synchronize the in-memory state of this bucket with its underlying storage.

        It clears the local cache and refresh the connection.
        This can be used to avoid connection timeout in long-running process.
        """
        resetCaches()
        # We need to refresh the connection to apply cache reset
        self._connection.close()
        self._db.pool.clear()  # <= By default, connection are reused, this ensure we create a fresh connection
        self._connection = self._get_connection()

    def _update(self, trees: Iterable[Tree], commit: bool) -> None:
        if commit:
            with self.transaction():
                self._data.update({tree.oid.bytes: tree for tree in trees})

        else:
            self._data.update({tree.oid.bytes: tree for tree in trees})
            self._savepoint()

    def add(self, tree: Tree) -> None:
        """Add a single :py:class:`~architxt.tree.Tree` to the bucket."""
        self._data[tree.oid.bytes] = tree

    def discard(self, tree: Tree) -> None:
        """Remove a :py:class:`~architxt.tree.Tree` from the bucket if it exists."""
        self._data.pop(tree.oid.bytes, None)

    def clear(self) -> None:
        """Remove all :py:class:`~architxt.tree.Tree` objects from the bucket."""
        self._data.clear()
        self._savepoint()

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
        return iter(self._data.values())

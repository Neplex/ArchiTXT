from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Generator, Iterable, MutableSet
from contextlib import nullcontext
from typing import TYPE_CHECKING, overload

import more_itertools
from aiostream import stream
from typing_extensions import Self

from architxt.tree import Forest, Tree, TreeOID
from architxt.utils import get_commit_batch_size

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from types import TracebackType

__all__ = ['TreeBucket']


class TreeBucket(ABC, MutableSet[Tree], Forest):
    """
    Abstract base class for a scalable, persistent, transactional container of :py:class:`~architxt.tree.Tree` objects.

    ``TreeBucket`` behaves like a :py:class:`set` while providing durable storage and explicit transactional
    semantics. It is designed for large-scale data and supports millions of trees with bounded
    memory usage through batched commits.

    **Transaction Management**

    - Adding or removing trees requires an active transaction.
    - Modifying a tree that is already in a bucket is not possible without a transaction.
      The modifications are automatically persisted when the transaction is committed.
    - Exceptions inside a ``with bucket.transaction():`` block automatically roll back the transaction.

    **Available Implementations**

    .. inheritance-diagram:: architxt.bucket.TreeBucket
        :include-subclasses:
        :parts: 1

    """

    def _update(self, trees: Iterable[Tree], commit: bool) -> None:
        with self.transaction() if commit else nullcontext():
            for tree in trees:
                self.add(tree)

    def update(self, trees: Iterable[Tree], *, commit: bool | int = False) -> None:
        """
        Add multiple :py:class:`~architxt.tree.Tree` to the bucket.

        :param trees: Trees to add to the bucket.
        :param commit: Commit automatically. If already in a transaction, no commit is applied.
            - If False (default), no commits are made, it relies on the current transaction.
            - If True, commits in batch.
            - If an integer, commits every N tree.
            To avoid memory issues, we recommend using incremental commit with large iterables.
        """
        batch_size = get_commit_batch_size(commit)

        for chunk in more_itertools.ichunked(trees, batch_size):
            self._update(chunk, bool(commit))

    async def async_update(self, trees: Iterable[Tree] | AsyncIterable[Tree], *, commit: bool | int = False) -> None:
        """
        Asynchronously add multiple :py:class:`~architxt.tree.Tree` to the bucket.

        This method mirrors the behavior of :py:meth:`~TreeBucket.update` but supports asynchronous iteration.

        :param trees: Trees to add to the bucket.
        :param commit: Commit automatically. If already in a transaction, no commit is applied.
            - If False (default), no commits are made, it relies on the current transaction.
            - If True, commits in batch.
            - If an integer, commits every N tree.
            To avoid memory issues, we recommend using incremental commit with large iterables.
        """
        batch_size = get_commit_batch_size(commit)

        async for chunk in stream.chunks(stream.iterate(trees), batch_size):
            self._update(chunk, bool(commit))

    @abstractmethod
    def close(self) -> None:
        """Close the underlying storage and release any associated resources."""

    @abstractmethod
    def transaction(self) -> AbstractContextManager[None]:
        """
        Return a context manager for managing a transaction.

        Upon exiting the context, the transaction is automatically committed.
        If an exception occurs within the context, the transaction is rolled back.

        Transactions are reentrant.
        """

    @abstractmethod
    def sync(self) -> None:
        """
        Synchronize the in-memory state of this bucket with its underlying storage.

        Implementations typically flush, refresh and/or invalidate caches and reload metadata so that subsequent
        operations reflect external changes. This operation may be expensive, so it should be called sparingly,
        but it is often required in concurrent environments (e.g., when using threads or subprocesses).
        """

    @abstractmethod
    def oids(self) -> Generator[TreeOID, None, None]:
        """Yield the object IDs (OIDs) of all trees stored in the bucket."""

    @overload
    def __getitem__(self, key: TreeOID) -> Tree: ...

    @overload
    def __getitem__(self, key: Iterable[TreeOID]) -> Iterable[Tree]: ...

    @abstractmethod
    def __getitem__(self, key: TreeOID | Iterable[TreeOID]) -> Tree | Iterable[Tree]:
        """
        Retrieve one or more :py:class:`~architxt.tree.Tree` by their OID(s).

        :param key: A single object ID or a collection of object IDs to retrieve.
        :return: A single :py:class:`~architxt.tree.Tree` or a collection of :py:class:`~architxt.tree.Tree` objects.
            - bucket[oid] -> tree
            - bucket[[oid1, oid2, ...]] -> [tree1, tree2, ...]
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

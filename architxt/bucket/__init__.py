from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Generator, Iterable, MutableSet
from contextlib import AbstractContextManager
from types import TracebackType
from typing import overload

import anyio.to_thread
from aiostream import Stream, stream

from architxt.tree import Forest, Tree, TreeOID
from architxt.utils import BATCH_SIZE

__all__ = ['TreeBucket']


class TreeBucket(ABC, MutableSet[Tree], Forest):
    """
    A scalable, persistent, set-like container for :py:class:`~architxt.tree.Tree`.

    The :py:class:`TreeBucket` behaves like a mutable set and provides persistent storage.
    It is designed to handle large collections of trees efficiently,
    supporting standard set operations and transactional updates.

    **Transaction Management**

    - Bucket automatically handles transactions when adding or removing :py:class:`~architxt.tree.Tree` from the bucket.
    - If a :py:class:`~architxt.tree.Tree` is modified after being added to the bucket, you must call :py:meth:`~TreeBucket.commit` to persist those changes.

    **Available Implementations**

    .. inheritance-diagram:: architxt.bucket.TreeBucket architxt.bucket.zodb.ZODBTreeBucket
        :top-classes: architxt.bucket.TreeBucket
        :parts: 1

    """

    @abstractmethod
    def update(self, trees: Iterable[Tree], batch_size: int = BATCH_SIZE) -> None:
        """
        Add multiple :py:class:`~architxt.tree.Tree` to the bucket, managing memory via chunked transactions.

        :param trees: Trees to add to the bucket.
        :param batch_size: The number of trees to be added at once.
        """

    async def async_update(self, trees: AsyncIterable[Tree], batch_size: int = BATCH_SIZE) -> None:
        """
        Asynchronously add multiple :py:class:`~architxt.tree.Tree` to the bucket.

        This method mirrors the behavior of :py:meth:`~TreeBucket.update` but supports asynchronous iteration.
        Internally, it delegates each chunk to a background thread.

        :param trees: Trees to add to the bucket.
        :param batch_size: The number of trees to be added at once.
        """
        chunk_stream: Stream[list[Tree]] = stream.chunks(trees, batch_size)
        chunk: list[Tree]

        async with chunk_stream.stream() as streamer:
            async for chunk in streamer:
                await anyio.to_thread.run_sync(self.update, chunk)

    @abstractmethod
    def close(self) -> None:
        """Close the underlying storage and release any associated resources."""

    @abstractmethod
    def transaction(self) -> AbstractContextManager[None]:
        """
        Return a context manager for managing a transaction.

        Upon exiting the context, the transaction is automatically committed.
        If an exception occurs within the context, the transaction is rolled back.
        """

    @abstractmethod
    def commit(self) -> None:
        """Persist any in-memory changes to :py:class:`~architxt.tree.Tree` in the bucket."""

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

    def __enter__(self) -> 'TreeBucket':
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

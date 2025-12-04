from __future__ import annotations

import sys
from random import randrange
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import more_itertools

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

__all__ = ['BATCH_SIZE', 'ExceptionGroup', 'get_commit_batch_size', 'update_url_queries', 'windowed_shuffle']

BATCH_SIZE = 1024
T = TypeVar('T')


if sys.version_info < (3, 11):

    class ExceptionGroup(BaseException):
        def __init__(self, message: str, exceptions: Sequence[BaseException]) -> None:
            message += '\n'.join(f'  ({i}) {exc!r}' for i, exc in enumerate(exceptions, 1))
            super().__init__(message)

else:
    from builtins import ExceptionGroup


def update_url_queries(url: str, **p: Any) -> str:
    """
    Update query parameters in a URL.

    Merges existing query parameters with provided keyword arguments.
    If a parameter already exists, it will be overwritten.

    >>> update_url_queries('https://example.com?foo=1', bar='2')
    'https://example.com?foo=1&bar=2'

    >>> update_url_queries('https://example.com?foo=1', foo='overwritten')
    'https://example.com?foo=overwritten'

    :param url: The URL to update.
    :param p: Query parameters to add or update.
    :return: The URL with updated query parameters.
    """
    u = urlparse(url)
    q = dict(parse_qsl(u.query))
    q.update(p)
    return urlunparse(u._replace(query=urlencode(q)))


def get_commit_batch_size(commit: bool | int) -> int:
    """
    Derive the batch size for commit operations.

    :param commit: Commit mode.
        - If True or False, returns the default BATCH_SIZE.
        - If a positive integer, returns that value as the batch size.
    :return: The batch size to use for chunked operations.
    :raises ValueError: If commit is a non-positive integer.
    """
    if isinstance(commit, bool):
        batch_size = BATCH_SIZE
    elif commit > 0:
        batch_size = commit
    else:
        msg = 'Commit should be a boolean or a positive integer'
        raise ValueError(msg)

    return batch_size


def windowed_shuffle(iterable: Iterable[T], window_size: int = 10) -> Generator[T, None, None]:
    """
    Shuffle an :py:class:`~Iterable` by yielding items in a randomized order using a sliding window buffer.

    :param iterable: Iterable to shuffle.
    :param window_size: Size of the sliding window buffer.

    :yield: Shuffled items.
    :raise ValueError: If window_size is <= 1.
    """
    if window_size <= 1:
        msg = "window_size must be > 1"
        raise ValueError(msg)

    it = iter(iterable)
    buf = list(more_itertools.take(window_size, it))

    for item in it:
        idx = randrange(len(buf))
        yield buf.pop(idx)
        buf.append(item)

    while buf:
        idx = randrange(len(buf))
        yield buf.pop(idx)

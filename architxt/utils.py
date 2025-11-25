from __future__ import annotations

import sys
from random import randrange
from typing import TYPE_CHECKING, TypeVar

import more_itertools
import psutil

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

__all__ = ['BATCH_SIZE', 'ExceptionGroup', 'is_memory_low', 'windowed_shuffle']

BATCH_SIZE = 1024
T = TypeVar('T')


if sys.version_info < (3, 11):

    class ExceptionGroup(BaseException):
        def __init__(self, message: str, exceptions: Sequence[BaseException]) -> None:
            message += '\n'.join(f'  ({i}) {exc!r}' for i, exc in enumerate(exceptions, 1))
            super().__init__(message)

else:
    from builtins import ExceptionGroup


def is_memory_low(threshold_mb: int) -> bool:
    """Check if available system memory is below the specified threshold in MB."""
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    return available_memory < threshold_mb


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

import sys
from collections.abc import Generator, Iterable, Sequence
from random import shuffle
from typing import TypeVar

import more_itertools
import psutil

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
    buf = list(more_itertools.take(window_size, iterable))
    shuffle(buf)

    for item in iterable:
        yield buf.pop()
        buf.append(item)
        shuffle(buf)

    while buf:
        yield buf.pop()

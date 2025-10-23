import sys
from collections.abc import Generator, Iterable, Sequence
from random import randrange
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
    """
    Check whether the system's available memory is below the given threshold in megabytes.
    
    Parameters:
        threshold_mb (int): Threshold value in megabytes to compare against available memory.
    
    Returns:
        bool: `True` if available memory is less than `threshold_mb`, `False` otherwise.
    """
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    return available_memory < threshold_mb


def windowed_shuffle(iterable: Iterable[T], window_size: int = 10) -> Generator[T, None, None]:
    """
    Yield items from `iterable` in a randomized order using a sliding window buffer.
    
    Parameters:
        iterable (Iterable[T]): Source of items to shuffle.
        window_size (int): Size of the sliding buffer used for local shuffling; must be greater than 1.
    
    Returns:
        Generator[T, None, None]: Items from `iterable` produced in a windowed-random order.
    
    Raises:
        ValueError: If `window_size` is less than or equal to 1.
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
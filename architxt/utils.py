import sys
from collections.abc import Sequence

import psutil

__all__ = ['BATCH_SIZE', 'ExceptionGroup', 'is_memory_low']

BATCH_SIZE = 1024


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

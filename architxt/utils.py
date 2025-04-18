import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from cloudpickle import cloudpickle

if TYPE_CHECKING:
    from architxt.tree import Forest

__all__ = ['BATCH_SIZE', 'is_memory_low', 'read_cache', 'write_cache']

BATCH_SIZE = 1024


async def write_cache(forest: 'Forest', path: Path) -> None:
    with path.open('wb') as cache_file:
        await asyncio.to_thread(cloudpickle.dump, forest, cache_file, protocol=5, buffer_callback=None)


async def read_cache(path: Path) -> 'Forest':
    with path.open('rb') as cache_file:
        return await asyncio.to_thread(cloudpickle.load, cache_file)


def is_memory_low(threshold_mb: int) -> bool:
    """Check if available system memory is below the specified threshold in MB."""
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    return available_memory < threshold_mb

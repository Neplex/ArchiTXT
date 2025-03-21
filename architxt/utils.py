import asyncio
from pathlib import Path

from cloudpickle import cloudpickle

from architxt.tree import Forest

__all__ = ['read_cache', 'write_cache']


async def write_cache(forest: Forest, path: Path) -> None:
    with path.open('wb') as cache_file:
        await asyncio.to_thread(cloudpickle.dump, forest, cache_file, protocol=5, buffer_callback=None)


async def read_cache(path: Path) -> Forest:
    with path.open('rb') as cache_file:
        return await asyncio.to_thread(cloudpickle.load, cache_file)

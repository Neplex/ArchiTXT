import psutil

__all__ = ['BATCH_SIZE', 'is_memory_low']

BATCH_SIZE = 1024


def is_memory_low(threshold_mb: int) -> bool:
    """Check if available system memory is below the specified threshold in MB."""
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    return available_memory < threshold_mb

from collections.abc import Sequence
from typing import Any

import numpy as np
from architxt.similarity import TreeCluster, TreeClusterer
from architxt.tree import Tree

__all__ = ['create_test_clusterer']


def create_test_clusterer(clusters: dict[str, Sequence[Tree]], **kwargs: Any) -> TreeClusterer:
    """
    Create a TreeClusterer with pre-populated test clusters.

    This helper bypasses the normal fit() process and directly sets clusters for deterministic testing.
    All trees are assigned uniform probabilities (1.0).

    For testing purposes only.
    """
    clusterer = TreeClusterer(**kwargs)
    clusterer._clusters = {
        name: TreeCluster(
            trees=trees,
            probabilities=np.asarray([1.0] * len(trees), dtype=np.float64),
        )
        for name, trees in clusters.items()
    }
    return clusterer

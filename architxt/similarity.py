from __future__ import annotations

import contextlib
import dataclasses
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from itertools import combinations
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
from hdbscan import HDBSCAN
from Levenshtein import jaro_winkler
from Levenshtein import ratio as levenshtein_ratio
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from architxt.bucket import TreeBucket
from architxt.tree import Forest, NodeType, Tree, TreeOID, TreePersistentRef, has_type

__all__ = [
    'DECAY',
    'METRIC_FUNC',
    'TreeCluster',
    'TreeClusterView',
    'TreeClusterer',
    'entity_labels',
    'jaccard',
    'jaro',
    'levenshtein',
    'similarity',
]

MAX_SIM_CTX_DEPTH = 5
DECAY = 2
METRIC_FUNC = Callable[[Collection[str], Collection[str]], float]


def jaccard(x: Collection[str], y: Collection[str]) -> float:
    """
    Jaccard similarity.

    :param x: The first sequence of strings.
    :param y: The second sequence of strings.
    :return: The Jaccard similarity as a float between 0 and 1, where 1 means identical sequences.

    >>> jaccard({"A", "B"}, {"A", "B", "C"})
    0.6666666666666666

    >>> jaccard({"apple", "banana", "cherry"}, {"apple", "cherry", "date"})
    0.5

    >>> jaccard(set(), set())
    1.0

    """
    x_set = set(x)
    y_set = set(y)
    return len(x_set & y_set) / len(x_set | y_set) if x_set or y_set else 1.0


def levenshtein(x: Collection[str], y: Collection[str]) -> float:
    """Levenshtein similarity."""
    return levenshtein_ratio(sorted(x), sorted(y))


def jaro(x: Collection[str], y: Collection[str]) -> float:
    """Jaro winkler similarity."""
    return jaro_winkler(sorted(x), sorted(y))


DEFAULT_METRIC: METRIC_FUNC = jaro  # jaccard, levenshtein, jaro


def _validate_tau(tau: float) -> None:
    if not (0.0 <= tau <= 1.0):
        msg = "tau must be between 0 and 1"
        raise ValueError(msg)


def _validate_decay(decay: float) -> None:
    if decay <= 0:
        msg = "decay must be a positive number"
        raise ValueError(msg)


def similarity(
    x: Tree,
    y: Tree,
    *,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    decay: float = DECAY,
    max_sim_ctx_depth: int = MAX_SIM_CTX_DEPTH,
) -> float:
    r"""
    Compute the similarity between two tree objects based on their entity labels and context.

    The function uses a specified metric (such as Jaccard, Levenshtein, or Jaro-Winkler) to calculate the
    similarity between the labels of entities in the trees. The similarity is computed as a recursive weighted
    mean for each tree anestor, where the weight decays with the distance from the tree.

    .. math::
        \text{similarity}_\text{metric}(x, y) =
        \frac{\sum_{i=0}^{d_{\min}} \text{decay}^{-i} \cdot \text{metric}(P^x_i, P^y_i)}
             {\sum_{i=0}^{d_{\min}} \text{decay}^{-i}}

    where :math:`P^x_i` and :math:`P^y_i` are the :math:`i^\text{th}` parent nodes of
    :math:`x` and :math:`y` respectively, and :math:`d_{\\min}` is the depth of the shallowest tree
    from :math:`x` and :math:`y` up to the root (or a fixed maximum depth of `max_sim_ctx_depth`).

    :param x: The first tree object.
    :param y: The second tree object.
    :param metric: A metric function to compute the similarity between the entity labels of the two trees.
    :param decay: The decay factor for the weighted mean. Must be strictly greater than 0.
        The higher the value, the more the weight of context decreases with distance.
    :param max_sim_ctx_depth: The maximum depth of context to consider when computing similarity.
    :return: A similarity score between 0 and 1, where 1 indicates maximum similarity.

    >>> from architxt.tree import Tree
    >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
    >>> similarity(t[0], t[1], metric=jaccard)
    0.5555555555555555

    """
    _validate_decay(decay)

    # Subtrees with disjoint entities cannot be similar
    if x.entity_labels().isdisjoint(y.entity_labels()):
        return 0.0

    # If the trees are identical (same oid) or structurally equivalent (same label),
    # they are considered perfectly similar: no need to traverse ancestors
    if x.oid == y.oid or (has_type(x) and has_type(y) and x.label == y.label):
        return 1.0

    _x: Tree | None = x
    _y: Tree | None = y

    weight_sum = 0.0
    sim_sum = 0.0
    distance = 0

    while _x is not None and _y is not None and distance < max_sim_ctx_depth:
        # Strong equivalence at current level: same node instance or same syntactic label
        if _x.oid == _y.oid or (has_type(_x) and has_type(_y) and _x.label == _y.label):
            tree_sim = 1.0

        else:
            x_labels = _x.entity_labels()
            y_labels = _y.entity_labels()
            tree_sim = metric(x_labels, y_labels)

        # Calculate similarity for current level and accumulate weighted sum
        weight = decay ** (-distance)
        weight_sum += weight
        sim_sum += weight * tree_sim

        # Move to parent nodes
        _x = _x.parent
        _y = _y.parent
        distance += 1

    return np.clip(sim_sum / weight_sum, 0.0, 1.0)  # Needed to fix float issues


@dataclasses.dataclass(frozen=True)
class TreeCluster:
    trees: Sequence[Tree | TreePersistentRef]
    probabilities: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.trees) != len(self.probabilities):
            msg = "trees and probabilities must have the same length"
            raise ValueError(msg)


class TreeClusterView(Mapping[str, Sequence[Tree]]):
    def __init__(self, clusterer: TreeClusterer) -> None:
        self._clusterer = clusterer

    def __getitem__(self, key: str) -> Sequence[Tree]:
        return tuple(
            tree
            for tree_ref in self._clusterer.get_cluster(key).trees
            if (tree := self._clusterer._get_tree(tree_ref)) is not None
        )

    def __len__(self) -> int:
        return len(self._clusterer.get_clusters_keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._clusterer.get_clusters_keys())


class TreeClusterer:
    _clusterer: HDBSCAN
    _clusters: dict[str, TreeCluster]
    _bucket: TreeBucket | None

    def __init__(
        self,
        tau: float = 0.7,
        decay: float = DECAY,
        metric: METRIC_FUNC = DEFAULT_METRIC,
        max_sim_ctx_depth: int = MAX_SIM_CTX_DEPTH,
        max_height: int = 5,
        min_cluster_size: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Cluster similar subtrees in a forest using HDBSCAN on a custom entity-label and context-aware similarity metric.

        The primary use cases are:
        - Discovering equivalent syntactic contexts for entities (e.g., for coreference or normalization).
        - Identifying frequent patterns in tree-structured data.

        :param tau: The similarity threshold for clustering.
        :param metric: The similarity metric function used to compute the similarity between subtrees.
        :param decay: The similarity decay factor.
            The higher the value, the more the weight of context decreases with distance.
        :param max_sim_ctx_depth: The maximum depth of context to consider when computing similarity.
        :param max_height: The maximum height of subtrees to consider for clustering.
        :param min_cluster_size: The minimum size of a cluster
        """
        _validate_tau(tau)
        _validate_decay(decay)

        self._tau = tau
        self._decay = decay
        self._metric = metric
        self._max_sim_ctx_depth = max_sim_ctx_depth
        self._max_height = max_height
        self._clusterer = HDBSCAN(
            metric='precomputed',
            cluster_selection_epsilon=1 - self._tau,
            min_cluster_size=min_cluster_size,
            **kwargs,
        )
        self._clusters = {}
        self._bucket = None

    @property
    def clusters(self) -> Mapping[str, Sequence[Tree]]:
        return TreeClusterView(self)

    def fit_predict(self, forest: Forest | TreeBucket, **kwargs: Any) -> Mapping[str, Sequence[Tree]]:
        self.fit(forest, **kwargs)
        return self.clusters

    def fit(self, forest: Forest | TreeBucket, _all_subtrees: bool = True) -> None:
        """
        Cluster subtrees of a given tree based on their similarity.

        The clusters are created by applying a distance threshold `tau` to the linkage matrix
        which is derived from pairwise subtree similarity calculations.
        Subtrees that are similar enough (based on `tau` and the `metric`) are grouped into clusters.
        Each cluster is represented as a tuple of subtrees.

        :param forest: The forest from which to extract and cluster subtrees.
        :param _all_subtrees: If true, compute the similarity between all subtrees, else only the given trees are compared.
        :return: A set of tuples, where each tuple represents a cluster of subtrees that meet the similarity threshold.
        """
        self._clusters.clear()
        self._bucket = None

        if isinstance(forest, TreeBucket):
            self._bucket = forest

        subtrees = (
            tuple(
                subtree if self._bucket is None else self._bucket.get_persistent_ref(subtree)
                for tree in forest
                for subtree in tree.subtrees(
                    lambda x: (
                        x.height <= self._max_height and not has_type(x, NodeType.ENT) and not x.has_duplicate_entity()
                    )
                )
            )
            if _all_subtrees
            else tuple(tree if self._bucket is None else self._bucket.get_persistent_ref(tree) for tree in forest)
        )

        if len(subtrees) < 2:
            return

        # Compute distance matrix for all subtrees
        dist_matrix = self._compute_dist_matrix(subtrees)

        # HDBSCAN does not like zero distances, so we add a bit of jitter
        # We use a tiny uniform noise on exact zeros to breaks ties without changing semantics
        zeros_mask = dist_matrix == 0.0
        if np.any(zeros_mask):
            rng = np.random.default_rng()
            jitter = rng.uniform(1e-12, 1e-10, size=dist_matrix.shape)
            dist_matrix[zeros_mask] += jitter[zeros_mask]

        # Perform hierarchical clustering based on the distance threshold tau
        labels = self._clusterer.fit_predict(squareform(dist_matrix))

        # Group subtrees by cluster ID
        subtree_clusters = defaultdict(list)
        for idx, cluster_id in enumerate(labels):
            if cluster_id != -1:
                subtree_clusters[cluster_id].append(idx)

        # Sort clusters based on the HDBSCAN membership probability within the cluster.
        for cluster_num, cluster_indices in enumerate(subtree_clusters.values()):
            if len(cluster_indices) == 1:
                continue

            # Sort the cluster based on their membership probability
            cluster_probabilities = self._clusterer.probabilities_
            sorted_indices = sorted(cluster_indices, key=lambda i: cluster_probabilities[i], reverse=True)
            tree_cluster = TreeCluster(
                trees=tuple(subtrees[i] for i in sorted_indices),
                probabilities=np.asarray(cluster_probabilities[sorted_indices], dtype=np.float64),
            )

            # Get the most common label for the cluster
            label_counter = Counter(
                tree.label.name for i in sorted_indices if has_type(tree := self._get_tree(subtrees[i]))
            )

            cluster_name = str(cluster_num)
            if most_commons := label_counter.most_common(1):
                cluster_name = f'{most_commons[0][0]}_{cluster_name}'

            self._clusters[cluster_name] = tree_cluster

    def get_equiv_of(self, t: Tree, top_k: int | None = 20) -> str | None:
        """
        Get the cluster containing the specified tree `t` based on similarity comparisons with the given set of clusters.

        The clusters are assessed using the provided similarity metric and threshold `tau`.

        :param t: The tree from which to extract and cluster subtrees.
        :param top_k: Compute the similarity only against the `top_k` elements.
            If `None` compute it against every element of the clusters.
        :return: The name of the cluster that meets the similarity threshold.
        :raises ValueError: If clusters have not been computed yet.
        """
        distance_to_center: dict[str, float] = {}
        for cluster_name, cluster in self._clusters.items():
            # fast membership check by ref
            if self._bucket is None:
                if t in cluster.trees:
                    return cluster_name

            else:
                with contextlib.suppress(KeyError):  # KeyError: t is not in the bucket
                    t_ref = self._bucket.get_persistent_ref(t)
                    if t_ref in cluster.trees:
                        return cluster_name

            center = next((tree for tree_ref in cluster.trees if (tree := self._get_tree(tree_ref)) is not None), None)
            if not center:
                # If all tree in the cluster does not exist anymore, we simply skip this cluster
                continue

            cluster_sim = self._similarity(t, center)
            if cluster_sim >= self._tau:
                return cluster_name

            distance_to_center[cluster_name] = cluster_sim

        # Sort equiv subtrees by similarity to the center element (the first one as the cluster are sorted)
        sorted_equiv_subtrees = sorted(distance_to_center.items(), key=lambda x: x[1], reverse=True)

        for cluster_name, _ in sorted_equiv_subtrees:
            # Early exit: stop checking once we find a matching cluster
            if self._get_cluster_similarities(t, cluster_name, top_k) >= self._tau:
                return cluster_name

        # No similar cluster found
        return None

    def _get_cluster_similarities(self, t: Tree, cluster_name: str, top_k: int | None) -> float:
        cluster = self._clusters[cluster_name]
        similarity_sum = 0.0
        weight_sum = 0.0

        for i, (tree_ref, prob) in enumerate(zip(cluster.trees, cluster.probabilities)):
            if (tree := self._get_tree(tree_ref)) is None:
                continue

            similarity_sum += self._similarity(t, tree) * prob
            weight_sum += prob

            if top_k is not None and i >= top_k:
                break

        return similarity_sum / weight_sum

    def get_cluster(self, key: str) -> TreeCluster:
        return self._clusters[key]

    def get_clusters_keys(self) -> set[str]:
        return set(self._clusters.keys())

    def _similarity(self, x: Tree, y: Tree) -> float:
        return similarity(x, y, metric=self._metric, decay=self._decay, max_sim_ctx_depth=self._max_sim_ctx_depth)

    def _get_tree(self, x: Tree | TreePersistentRef) -> Tree | None:
        if isinstance(x, Tree):
            return x

        if self._bucket is not None:
            try:
                return self._bucket.resolve_ref(x)
            except KeyError:
                return None

        msg = "Cannot resolve TreePersistentRef without a TreeBucket."
        raise ValueError(msg)

    @staticmethod
    def _tree_dist(x: Tree, y: Tree) -> int:
        return min(
            abs(x.height - y.height),
            abs(x.depth - y.depth),
        )

    def _compute_dist_matrix(self, subtrees: Collection[Tree | TreePersistentRef]) -> npt.NDArray[np.double]:
        """
        Compute the condensed distance matrix for a collection of subtrees.

        This function computes pairwise distances between all subtrees and stores the results
        in a condensed distance matrix format (1D array), which is suitable for hierarchical clustering.

        The computation is sequential.

        :param subtrees: A list of subtrees reference for which pairwise distances will be calculated.
        :return: A 1D numpy array containing the condensed distance matrix (only a triangle of the full matrix).
        """
        nb_combinations = math.comb(len(subtrees), 2)
        dist_matrix = np.full((nb_combinations,), 1.0, dtype=np.double)

        tree_pairs = combinations(subtrees, 2)
        tree_pairs = tqdm(tree_pairs, desc='similarity', total=nb_combinations, leave=False, unit_scale=True)

        for i, (x_ref, y_ref) in enumerate(tree_pairs):
            x = self._get_tree(x_ref)
            y = self._get_tree(y_ref)

            if x is None or y is None:
                continue

            if self._tree_dist(x, y) < self._max_height:
                dist_matrix[i] = 1 - self._similarity(x, y)

        return dist_matrix

    def mlflow_plot(self, base_path: str) -> None:
        """
        Plot clustering result as mlflow artifacts.

        :param base_path: The base path where to save the artifacts.
        """
        try:
            ax = self._clusterer.condensed_tree_.plot(select_clusters=True, label_clusters=True)
            mlflow.log_figure(ax.get_figure(), f'{base_path}/condensed_tree.svg')
            plt.close(ax.get_figure())
        except Exception as error:
            warnings.warn(f'Could not plot condensed tree: {error}', stacklevel=2)

        try:
            ax = self._clusterer.single_linkage_tree_.plot()
            mlflow.log_figure(ax.get_figure(), f'{base_path}/single_linkage_tree.svg')
            plt.close(ax.get_figure())
        except Exception as error:
            warnings.warn(f'Could not plot single linkage tree: {error}', stacklevel=2)


def entity_labels(
    forest: Iterable[Tree],
    *,
    tau: float,
    metric: METRIC_FUNC | None = DEFAULT_METRIC,
    decay: float = DECAY,
) -> dict[TreeOID, str]:
    """
    Process the given forest to assign labels to entities based on clustering of their ancestor.

    :param forest: The forest from which to extract and cluster entities.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
        If None, use the parent label as the equivalent class.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :return: A dictionary mapping entities to their respective cluster name.
    """
    _validate_tau(tau)

    if metric is None:
        return {entity.oid: entity.parent.label for tree in forest for entity in tree.entities() if entity.parent}

    entity_parents = [
        subtree
        for tree in forest
        for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and x.has_entity_child())
    ]
    clusterer = TreeClusterer(tau=tau, metric=metric, decay=decay)
    clusterer.fit(entity_parents, _all_subtrees=False)

    return {
        child.oid: cluster_name
        for cluster_name, cluster in clusterer.clusters.items()
        for subtree in cluster
        for child in subtree
        if has_type(child, NodeType.ENT)
    }

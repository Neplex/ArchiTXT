import math
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations

import numpy as np
import numpy.typing as npt
from Levenshtein import jaro_winkler
from Levenshtein import ratio as levenshtein_ratio
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from tqdm import tqdm

from architxt.model import NodeType
from architxt.tree import Forest, Tree, has_type

METRIC_FUNC = Callable[[set[str], set[str]], float]
TREE_CLUSTER = set[tuple[Tree, ...]]


def jaccard(x: set[str], y: set[str]) -> float:
    """
    Jaccard similarity

    :param x: The first sequence of strings.
    :param y: The second sequence of strings.
    :return: The Jaccard similarity as a float between 0 and 1, where 1 means identical sequences.

    Example:
    >>> jaccard({"A", "B"}, {"A", "B", "C"})
    0.6666666666666666

    >>> jaccard({"apple", "banana", "cherry"}, {"apple", "cherry", "date"})
    0.5

    >>> jaccard(set(), set())
    1.0
    """
    return len(x & y) / len(x | y) if x or y else 1.0


def levenshtein(x: set[str], y: set[str]) -> float:
    """
    Levenshtein similarity
    """
    return levenshtein_ratio(sorted(x), sorted(y))


def jaro(x: set[str], y: set[str]) -> float:
    """
    Jaro winkler similarity
    """
    return jaro_winkler(sorted(x), sorted(y))


DEFAULT_METRIC: METRIC_FUNC = jaro  # jaccard, levenshtein, jaro


def similarity(x: Tree, y: Tree, *, metric: METRIC_FUNC = DEFAULT_METRIC) -> float:
    """
    Computes the similarity between two tree objects based on their entity labels and context.
    The function uses a specified metric (such as Jaccard, Levenshtein, or Jaro-Winkler) to calculate the
    similarity between the labels of entities in the trees. The similarity is computed as recursive weighted
    mean for each tree anestor.

    :param x: The first tree object.
    :param y: The second tree object.
    :param metric: A metric function to compute the similarity between the entity labels of the two trees.
    :return: A similarity score between 0 and 1, where 1 indicates maximum similarity.

    Example:
    >>> from architxt.tree import Tree
    >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
    >>> similarity(t[0], t[1], metric=jaccard)
    0.5555555555555555
    """
    assert x is not None
    assert y is not None

    if x is y:
        return 1.0

    weight_sum = 0.0
    sim_sum = 0.0
    distance = 1

    while x is not None and y is not None:
        # Extract the entity labels as sets for faster lookup
        x_labels = x.entity_labels()
        y_labels = y.entity_labels()

        # If no common entity labels, return similarity 0 early
        if x_labels.isdisjoint(y_labels):
            return 0.0

        # Calculate similarity for current level and accumulate weighted sum
        weight = 1 / distance
        weight_sum += weight
        sim_sum += weight * metric(x_labels, y_labels)

        # Move to parent nodes
        x = x.parent()
        y = y.parent()
        distance += 1

    return min(max(sim_sum / weight_sum, 0), 1)  # Need to fix float issues


def sim(x: Tree, y: Tree, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> bool:
    """
    Determines whether the similarity between two tree objects exceeds a given threshold `tau`.

    :param x: The first tree object to compare.
    :param y: The second tree object to compare.
    :param tau: The threshold value for similarity.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :return: `True` if the similarity between `x` and `y` is greater than or equal to `tau`, otherwise `False`.

    Example:
    >>> from architxt.tree import Tree
    >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
    >>> sim(t[0], t[1], tau=0.5, metric=jaccard)
    True
    """
    return similarity(x, y, metric=metric) >= tau


def compute_dist_matrix(subtrees: list[Tree], *, metric: METRIC_FUNC) -> npt.NDArray[np.uint16]:
    """
    Compute the condensed distance matrix for a collection of subtrees.

    This function computes pairwise distances between all subtrees and stores the results
    in a condensed distance matrix format (1D array), which is suitable for hierarchical clustering.

    The computation is sequential.

    :param subtrees: A list of subtrees for which pairwise distances will be calculated.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :return: A 1D numpy array containing the condensed distance matrix (only a triangle of the full matrix).
    """
    nb_combinations = math.comb(len(subtrees), 2)

    distances = (
        np.uint16((1 - similarity(x, y, metric=metric)) * 10000) if abs(x.height() - y.height()) < 5 else np.nan
        for x, y in combinations(subtrees, 2)
    )

    return np.fromiter(
        tqdm(
            distances,
            desc='similarity',
            total=nb_combinations,
            leave=False,
            unit_scale=True,
        ),
        count=nb_combinations,
        dtype=np.uint16,
    )


def equiv_cluster(
    trees: Forest, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC, _all_subtrees: bool = True
) -> TREE_CLUSTER:
    """
    Clusters subtrees of a given tree based on their similarity. The clusters are created by applying
    a distance threshold `tau` to the linkage matrix, which is derived from pairwise subtree similarity calculations.
    Subtrees that are similar enough (based on `tau` and the `metric`) are grouped into clusters. Each cluster
    is represented as a tuple of subtrees.

    :param trees: The forest from which to extract and cluster subtrees.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    :return: A set of tuples, where each tuple represents a cluster of subtrees that meet the similarity threshold.
    """
    subtrees = (
        [
            subtree
            for tree in trees
            for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and not x.has_duplicate_entity())
        ]
        if _all_subtrees
        else trees
    )

    if len(subtrees) < 2:
        return set()

    # Compute distance matrix for all subtrees
    dist_matrix = compute_dist_matrix(subtrees, metric=metric)

    # Perform hierarchical clustering based on the distance threshold tau
    linkage_matrix = hierarchy.linkage(dist_matrix, method='single')
    clusters = hierarchy.fcluster(linkage_matrix, 1 - tau, criterion='distance')

    # Group subtrees by cluster ID
    subtree_clusters = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        subtree_clusters[cluster_id].append(idx)

    # Sort clusters based on the center element (the closest subtree to all others)
    # We determine the center by computing the sum of distances for each subtree to all others in the cluster.
    # The index of the subtree with the smallest sum of distances is the center.
    square_dist_matrix = squareform(dist_matrix)
    sorted_clusters = set()

    for cluster_indices in subtree_clusters.values():
        sum_distances = np.sum(square_dist_matrix[np.ix_(cluster_indices, cluster_indices)], axis=1)
        center_index = cluster_indices[np.argmin(sum_distances)]

        # Sort the cluster based on distance to the center
        sorted_cluster = sorted(cluster_indices, key=lambda idx: square_dist_matrix[center_index][idx])

        # Add the sorted cluster as a tuple to the set (immutable and hashable)
        sorted_clusters.add(tuple(subtrees[i] for i in sorted_cluster))

    return sorted_clusters


def get_equiv_of(
    t: Tree, equiv_subtrees: TREE_CLUSTER, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC
) -> tuple[Tree, ...]:
    """
    Returns the cluster containing the specified tree `t` based on similarity comparisons
    with the given set of clusters. The clusters are assessed using the provided similarity
    metric and threshold `tau`.

    :param t: The tree from which to extract and cluster subtrees.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    :return: A tuple representing the cluster of subtrees that meet the similarity threshold.
    """
    distance_to_center = []
    for cluster in equiv_subtrees:
        if t in cluster or (cluster_sim := similarity(t, cluster[0], metric=metric)) >= tau:
            return cluster

        distance_to_center.append(cluster_sim)

    # Sort equiv subtrees by similarity to the center element (the first one as the cluster are sorted)
    equiv_subtrees = sorted(zip(equiv_subtrees, distance_to_center), key=lambda x: x[1], reverse=True)

    for cluster, _ in equiv_subtrees:
        # Early exit: stop checking once we find a matching cluster
        if t in cluster or any(sim(x, t, tau, metric) for x in cluster):
            return cluster

    # Return empty tuple if no similar cluster is found
    return ()


def entity_labels(forest: Forest, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> dict[str:int]:
    """
    Process the given forest to assign labels to entities based on clustering of their ancestor.

    :param forest: The forest from which to extract and cluster entities.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    :return: A dictionary mapping entities to their respective cluster IDs.
    """
    entity_parents = [
        subtree
        for tree in forest
        for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and x.has_entity_child())
    ]
    equiv_subtrees = equiv_cluster(entity_parents, tau=tau, metric=metric, _all_subtrees=False)

    return {
        f'{child.label().name}${' '.join(child)}': i
        for i, cluster in enumerate(equiv_subtrees)
        for subtree in cluster
        for child in subtree
        if has_type(child, NodeType.ENT)
    }

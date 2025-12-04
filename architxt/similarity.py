from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from itertools import combinations

import mlflow
import numpy as np
import numpy.typing as npt
import plotly.figure_factory as ff
from Levenshtein import jaro_winkler
from Levenshtein import ratio as levenshtein_ratio
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from architxt.tree import NodeType, Tree, TreeOID, has_type

MAX_DEPTH = 5
DECAY = 2
METRIC_FUNC = Callable[[Collection[str], Collection[str]], float]
TREE_CLUSTER = dict[str, Sequence[Tree]]


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


def similarity(x: Tree, y: Tree, *, metric: METRIC_FUNC = DEFAULT_METRIC, decay: float = DECAY) -> float:
    r"""
    Compute the similarity between two tree objects based on their entity labels and context.

    The function uses a specified metric (such as Jaccard, Levenshtein, or Jaro-Winkler) to calculate the
    similarity between the labels of entities in the trees. The similarity is computed as a recursive weighted
    mean for each tree anestor, where the weight decays with the distance from the tree.

    .. math::
        \text{similarity}_\text{metric}(x, y) =
        \frac{\sum_{i=1}^{d_{\min}} \text{decay}^{-i} \cdot \text{metric}(P^x_i, P^y_i)}
             {\sum_{i=1}^{d_{\min}} \text{decay}^{-i}}

    where :math:`P^x_i` and :math:`P^y_i` are the :math:`i^\text{th}` parent nodes of
    :math:`x` and :math:`y` respectively, and :math:`d_{\\min}` is the depth of the shallowest tree
    from :math:`x` and :math:`y` up to the root (or a fixed maximum depth).

    :param x: The first tree object.
    :param y: The second tree object.
    :param metric: A metric function to compute the similarity between the entity labels of the two trees.
    :param decay: The decay factor for the weighted mean. Must be strictly greater than 0.
        The higher the value, the more the weight of context decreases with distance.
    :return: A similarity score between 0 and 1, where 1 indicates maximum similarity.

    >>> from architxt.tree import Tree
    >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
    >>> similarity(t[0], t[1], metric=jaccard)
    0.5555555555555555

    """
    if decay <= 0:
        msg = "decay must be a positive number"
        raise ValueError(msg)

    if x.oid == y.oid or x.label == y.label:
        return 1.0

    _x: Tree | None = x
    _y: Tree | None = y

    weight_sum = 0.0
    sim_sum = 0.0
    distance = 1

    while _x is not None and _y is not None and distance <= MAX_DEPTH:
        # Extract the entity labels as sets for faster lookup
        x_labels = _x.entity_labels()
        y_labels = _y.entity_labels()

        # If no common entity labels, return similarity 0 early
        if x_labels.isdisjoint(y_labels):
            return 0.0

        # Calculate similarity for current level and accumulate weighted sum
        weight = decay ** (-distance)
        weight_sum += weight
        sim_sum += weight * metric(x_labels, y_labels)

        # Move to parent nodes
        _x = _x.parent
        _y = _y.parent
        distance += 1

    return min(max(sim_sum / weight_sum, 0), 1)  # Need to fix float issues


def sim(x: Tree, y: Tree, tau: float, *, metric: METRIC_FUNC = DEFAULT_METRIC, decay: float = DECAY) -> bool:
    """
    Determine whether the similarity between two tree objects exceeds a given threshold `tau`.

    :param x: The first tree object to compare.
    :param y: The second tree object to compare.
    :param tau: The threshold value for similarity.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :return: `True` if the similarity between `x` and `y` is greater than or equal to `tau`, otherwise `False`.

    >>> from architxt.tree import Tree
    >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
    >>> sim(t[0], t[1], tau=0.5, metric=jaccard)
    True

    """
    return similarity(x, y, metric=metric, decay=decay) >= tau


def compute_dist_matrix(
    subtrees: Collection[Tree], *, metric: METRIC_FUNC, decay: float = DECAY
) -> npt.NDArray[np.float32]:
    """
    Compute the condensed distance matrix for a collection of subtrees.

    This function computes pairwise distances between all subtrees and stores the results
    in a condensed distance matrix format (1D array), which is suitable for hierarchical clustering.

    The computation is sequential.

    :param subtrees: A list of subtrees for which pairwise distances will be calculated.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :return: A 1D numpy array containing the condensed distance matrix (only a triangle of the full matrix).
    """
    nb_combinations = math.comb(len(subtrees), 2)

    distances = (
        (1 - similarity(x, y, metric=metric, decay=decay)) if abs(x.height - y.height) < 5 else 1.0
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
        dtype=np.float32,
    )


def equiv_cluster(
    trees: Iterable[Tree],
    *,
    tau: float,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    decay: float = DECAY,
    _all_subtrees: bool = True,
    _step: int | None = None,
) -> TREE_CLUSTER:
    """
    Cluster subtrees of a given tree based on their similarity.

    The clusters are created by applying a distance threshold `tau` to the linkage matrix
    which is derived from pairwise subtree similarity calculations.
    Subtrees that are similar enough (based on `tau` and the `metric`) are grouped into clusters.
    Each cluster is represented as a tuple of subtrees.

    :param trees: The forest from which to extract and cluster subtrees.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :param _all_subtrees: If true, compute the similarity between all subtrees, else only the given trees are compared.
    :param _step: The MLFlow step for logging.
    :return: A set of tuples, where each tuple represents a cluster of subtrees that meet the similarity threshold.
    """
    subtrees = (
        [
            subtree
            for tree in trees
            for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and not x.has_duplicate_entity())
        ]
        if _all_subtrees
        else tuple(trees)
    )

    if len(subtrees) < 2:
        return {}

    # Compute distance matrix for all subtrees
    dist_matrix = compute_dist_matrix(subtrees, metric=metric, decay=decay)

    # Perform hierarchical clustering based on the distance threshold tau
    linkage_matrix = hierarchy.linkage(dist_matrix, method='single')
    clusters = hierarchy.fcluster(linkage_matrix, 1 - tau, criterion='distance')

    square_dist_matrix = squareform(dist_matrix)

    if mlflow.active_run() and _step is not None:
        labels = [st.label for st in subtrees]

        fig = ff.create_annotated_heatmap(z=square_dist_matrix, colorscale='Cividis', x=labels, y=labels)
        mlflow.log_figure(fig, f'similarity/{_step}/heatmap.html')

        fig = ff.create_dendrogram(
            linkage_matrix,
            orientation='left',
            color_threshold=1 - tau,
            labels=labels,
            linkagefun=lambda _: linkage_matrix,
        )
        mlflow.log_figure(fig, f'similarity/{_step}/dendrogram.html')

    # Group subtrees by cluster ID
    subtree_clusters = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        subtree_clusters[cluster_id].append(idx)

    # Sort clusters based on the center element (the closest subtree to all others)
    # We determine the center by computing the sum of distances for each subtree to all others in the cluster.
    # The index of the subtree with the smallest sum of distances is the center.
    sorted_clusters: TREE_CLUSTER = {}

    for cluster_num, cluster_indices in enumerate(subtree_clusters.values()):
        sum_distances = np.sum(square_dist_matrix[np.ix_(cluster_indices, cluster_indices)], axis=1)
        center_index = cluster_indices[np.argmin(sum_distances)]

        # Sort the cluster based on distance to the center
        sorted_cluster = tuple(
            subtrees[i] for i in sorted(cluster_indices, key=lambda i: square_dist_matrix[center_index][i])
        )

        # Get the most common label for the cluster
        cluster_name = str(cluster_num)
        if most_commons := Counter(tree.label.name for tree in sorted_cluster if has_type(tree)).most_common(1):
            cluster_name = f'{most_commons[0][0]}_{cluster_name}'

        sorted_clusters[cluster_name] = sorted_cluster

    return sorted_clusters


def get_equiv_of(
    t: Tree,
    equiv_subtrees: TREE_CLUSTER,
    *,
    tau: float,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    decay: float = DECAY,
) -> str | None:
    """
    Get the cluster containing the specified tree `t` based on similarity comparisons with the given set of clusters.

    The clusters are assessed using the provided similarity metric and threshold `tau`.

    :param t: The tree from which to extract and cluster subtrees.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :return: The name of the cluster that meets the similarity threshold.
    """
    distance_to_center = {}
    for cluster_name, cluster in equiv_subtrees.items():
        if t in cluster or (cluster_sim := similarity(t, cluster[0], metric=metric, decay=decay)) >= tau:
            return cluster_name

        distance_to_center[cluster_name] = cluster_sim

    # Sort equiv subtrees by similarity to the center element (the first one as the cluster are sorted)
    sorted_equiv_subtrees = sorted(distance_to_center.items(), key=lambda x: x[1], reverse=True)

    for cluster_name, _ in sorted_equiv_subtrees:
        cluster = equiv_subtrees[cluster_name]

        # Early exit: stop checking once we find a matching cluster
        if t in cluster or any(sim(x, t, tau=tau, metric=metric, decay=decay) for x in cluster):
            return cluster_name

    # Return an empty tuple if no similar cluster is found
    return None


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
    if metric is None:
        return {entity.oid: entity.parent.label for tree in forest for entity in tree.entities() if entity.parent}

    entity_parents = (
        subtree
        for tree in forest
        for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and x.has_entity_child())
    )
    equiv_subtrees: TREE_CLUSTER = equiv_cluster(
        entity_parents, tau=tau, metric=metric, decay=decay, _all_subtrees=False
    )

    return {
        child.oid: cluster_name
        for cluster_name, cluster in equiv_subtrees.items()
        for subtree in cluster
        for child in subtree
        if has_type(child, NodeType.ENT)
    }

import math
import multiprocessing
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import combinations

import more_itertools
import numpy as np
import numpy.typing as npt
import ray
from Levenshtein import jaro_winkler
from Levenshtein import ratio as levenshtein_ratio
from ray.experimental import tqdm_ray
from scipy.cluster import hierarchy
from tqdm import tqdm

from architxt.model import TREE_POS, NodeType
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
    if not x and not y:
        return 1.0

    x = set(x)
    y = set(y)

    return len(x & y) / len(x | y)


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


# @cached(cache=SIM_CACHE, lock=SIM_CACHE_LOCK, key=lambda x, y, *, metric: (x.treeposition(), y.treeposition()))
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


@ray.remote
def compute_distance(
    batch: Iterable[tuple[int, ray.ObjectRef, TREE_POS, ray.ObjectRef, TREE_POS]], *, metric: METRIC_FUNC, pbar
) -> list[tuple[int, np.uint16]]:
    """
    Compute the distance between two subtrees.

    The distance is based on a similarity measure (bounded between [0, 1]).
    Since exact float values are not required for hierarchical clustering,
    we transform the similarity score into an integer and invert it to create a
    distance metric, which is stored as an unsigned short (np.ushort) to minimize memory usage.

    :param batch: A, iterable of index and tree position pair to compute.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :return: The computed distance as an unsigned short integer, representing the scaled and inverted similarity score.
    """
    distances_idx = []

    for sub_batch in more_itertools.chunked(batch, 100):
        for idx, x_ref, x_pos, y_ref, y_pos in sub_batch:
            x, y = ray.get([x_ref, y_ref])
            distance = np.uint16((1 - similarity(x[x_pos], y[y_pos], metric=metric)) * 10000)
            distances_idx.append((idx, distance))
        pbar.update.remote(len(sub_batch))

    return distances_idx


def compute_dist_matrix(
    subtrees: list[Tree],
    *,
    metric: METRIC_FUNC,
    max_tasks: int | None = None,
    batch_size: int = 1_000_000,
) -> npt.NDArray[np.uint16]:
    """
    Compute the condensed distance matrix for a collection of subtrees.

    This function computes pairwise distances between all subtrees and stores the results
    in a condensed distance matrix format (1D array), which is suitable for hierarchical clustering.

    The computation is done in parallel using RAY.

    :param subtrees: A list of subtrees for which pairwise distances will be calculated.
    :param metric: A callable similarity metric to compute the similarity between the two trees.
    :param max_tasks: The maximum number of parallel Ray tasks to process at a time.
    :param batch_size: Number of pairs to compute in each batch.
    :return: A 1D numpy array containing the condensed distance matrix (only a triangle of the full matrix).
    """
    if not max_tasks:
        max_tasks = int(ray.available_resources().get('CPU', multiprocessing.cpu_count())) - 1

    # Get the reference to the root trees
    trees = list({subtree.root() for subtree in subtrees})
    trees_refs = [ray.put(tree) for tree in trees]

    # Prepare the pair combinations, we skip trees that are not close enough
    pair_combinations = (
        (idx, trees_refs[trees.index(x.root())], x.treeposition(), trees_refs[trees.index(y.root())], y.treeposition())
        for idx, (x, y) in enumerate(combinations(subtrees, 2))
        if abs(x.height() - y.height()) < 5
    )

    # Condensed distance matrix (1D array of the triangular part)
    nb_combinations = math.comb(len(subtrees), 2)
    dist_matrix = np.full(nb_combinations, np.nan, dtype=np.uint16)

    task_refs: list[ray.ObjectRef] = []

    # Iterate over combinations and launch tasks
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=nb_combinations, desc='similarity')

    for pair_batch in more_itertools.chunked(pair_combinations, batch_size):
        distance_ref = compute_distance.remote(pair_batch, metric=metric, pbar=pbar)
        task_refs.append(distance_ref)

        # Wait for tasks to complete when max_tasks is reached
        if len(task_refs) >= max_tasks:
            ready_refs, task_refs = ray.wait(task_refs, num_returns=1)
            for task in ray.get(ready_refs):
                for idx, distance in task:
                    dist_matrix[idx] = distance
                # pbar.update(batch_size)

    # Process remaining tasks
    for task in ray.get(task_refs):
        for idx, distance in task:
            dist_matrix[idx] = distance
        # pbar.update(batch_size)

    pbar.close.remote()

    return dist_matrix


def compute_dist_matrix_local(subtrees: list[Tree], *, metric: METRIC_FUNC) -> npt.NDArray[np.uint16]:
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


def equiv_cluster(trees: Forest, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> TREE_CLUSTER:
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
    subtrees = [
        subtree
        for tree in trees
        for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT) and not x.has_duplicate_entity())
    ]

    if len(subtrees) < 2:
        return set()

    # Compute distance matrix for all subtrees
    dist_matrix = compute_dist_matrix_local(subtrees, metric=metric)

    # Hierarchical clustering with average linkage
    linkage_matrix = hierarchy.linkage(dist_matrix, method='single')

    # Cluster based on the distance threshold tau
    clusters = hierarchy.fcluster(linkage_matrix, 1 - tau, criterion='distance')

    # Group subtrees into clusters
    subtree_clusters = defaultdict(set)
    for subtree, cluster_id in zip(subtrees, clusters, strict=False):
        subtree_clusters[cluster_id].add(subtree)

    # Return clusters as a set of tuples (immutable and hashable)
    return {tuple(cluster) for cluster in subtree_clusters.values()}


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
    :return: A set of tuples, where each tuple represents a cluster of subtrees that meet the similarity threshold.
    """
    for cluster in sorted(equiv_subtrees, key=len, reverse=True):
        if any(sim(x, t, tau, metric) for x in cluster):
            return cluster

    return ()

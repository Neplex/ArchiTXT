from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations

from Levenshtein import ratio as levenshtein_ratio, jaro_winkler
from cachetools import cached
from joblib import Parallel, delayed
from scipy.cluster import hierarchy
from tqdm import tqdm

from .tree import Tree, ParentedTree


def jaccard(x: list[str], y: list[str]) -> float:
    """
    Jaccard similarity for list (originally for set)

    Set similarity:
        len(x & y) / len(x | y)

    J(AAB, AB) = 1
    J_list(AAB, AB) = 2/3
    """
    counter_x = Counter(x)
    counter_y = Counter(y)

    inter = list((counter_x & counter_y).elements())
    union = list((counter_x | counter_y).elements())

    return len(inter) / len(union) if len(union) else 0


def levenshtein(x: list[str], y: list[str]) -> float:
    """
    Levenshtein similarity for list
    """
    return levenshtein_ratio(
        sorted(set(x)),
        sorted(set(y)),
    )


def jaro(x: list[str], y: list[str]) -> float:
    """
    Jaro winkler similarity for list
    """
    return jaro_winkler(
        sorted(x),
        sorted(y)
    )


METRIC_FUNC = Callable[[list[str], list[str]], float]
DEFAULT_METRIC: METRIC_FUNC = jaccard


@cached(cache={}, key=lambda x, y, *, metric: f'{metric.__name__}=={x.pformat()}#$#{y.pformat()}')
def similarity(x: ParentedTree, y: ParentedTree, *, metric: METRIC_FUNC = DEFAULT_METRIC) -> float:
    depth_max = max(x.depth(), y.depth())
    depth_min = min(x.depth(), y.depth())

    x_labels = tuple(node.label().name for node in x.entities())
    y_labels = tuple(node.label().name for node in y.entities())
    sim_sum = depth_min * metric(x_labels, y_labels)

    # Context similarity
    for i in range(1, depth_min):
        x = x.parent()
        y = y.parent()
        sim_sum += (depth_min - i) * similarity(x, y, metric=metric)

    weight_sum = sum(range(1, depth_min + 1))
    return sim_sum / weight_sum if weight_sum else 0


def sim(
        x: ParentedTree, y: ParentedTree,
        tau: float, metric: METRIC_FUNC = DEFAULT_METRIC
) -> bool:
    return similarity(x, y, metric=metric) >= tau


TREE_CLUSTER = set[tuple[ParentedTree, ...]]


def equiv_cluster(
        t: ParentedTree, *,
        tau: float, metric: METRIC_FUNC = DEFAULT_METRIC
) -> TREE_CLUSTER:
    """
    Clusters all subtrees of a given tree based on their similarity.
    The clusters are determined by applying a distance threshold, `tau`, to the linkage matrix obtained through the similarity function.
    Finally, the method returns a set of clusters, where each cluster is represented as a tuple of subtrees.

    :param t: the tree to works with
    :param tau: the threshold for clustering similarity
    :param metric: the similarity metric function
    :return: Set of tuples representing the clustered subtrees
    """
    subtrees = list(sorted(t.subtrees(
        lambda x: isinstance(x, Tree) and x != t.root() and max(Counter(ent.label() for ent in x.entities()).values()) == 1
    ), key=lambda x: x.height()))

    if len(subtrees) < 2:
        return []

    # Get distance matrix for all subtrees
    dist_matrix = Parallel(n_jobs=1, require='sharedmem')(
        delayed(lambda x, y: 1 - similarity(x, y, metric=metric))(x, y)
        for x, y in tqdm(list(combinations(subtrees, r=2)), desc='similarity', leave=False)
    )

    # with tqdm(desc='similarity', leave=False, total=int((N ** 2 - N) / 2)) as pb:
    #     dist_matrix = pdist(
    #         np.asarray(subtrees, dtype=Tree).reshape((len(subtrees), 1)),
    #         metric=lambda x, y: similarity(x[0], y[0], metric=metric, _pb=pb)
    #     )

    # Linkage matrix
    linkage_matrix = hierarchy.linkage(dist_matrix, method='average')

    # Cluster
    clusters = hierarchy.fcluster(linkage_matrix, 1 - tau, criterion='distance')

    subtree_clusters = defaultdict(list)
    for subtree, cluster in zip(subtrees, clusters):
        subtree_clusters[cluster].append(subtree)

    return {
        tuple(cluster)
        for cluster in subtree_clusters.values()
    }


def get_equiv_of(
        t: ParentedTree, equiv_subtrees: TREE_CLUSTER, *,
        tau: float, metric: METRIC_FUNC = DEFAULT_METRIC
) -> tuple[ParentedTree, ...] | None:
    """
    This function returns the cluster in which `t` belongs.
    :param t: the tree to check
    :param equiv_subtrees: the set of clusters
    :param tau: the threshold for clustering similarity
    :param metric: the similarity metric function
    :return: The cluster containing `t`
    """
    for cluster in tqdm(equiv_subtrees, desc='search equivalent class', leave=False):
        if all(sim(x, t, tau, metric) for x in cluster):
            return cluster

    return []

from apted import APTED
from apted import Config as APTEDConfig
from cachetools import cachedmethod

from .similarity import DEFAULT_METRIC, METRIC_FUNC, entity_labels, jaccard, similarity
from .tree import Forest, NodeType, Tree, has_type


class EditDistanceConfig(APTEDConfig):
    def rename(self, node1: Tree | str, node2: Tree | str) -> int:
        name1 = node1.label if isinstance(node1, Tree) else node1
        name2 = node2.label if isinstance(node2, Tree) else node2
        return int(name1 != name2)

    def children(self, node: Tree | str) -> list[Tree]:
        return node if isinstance(node, Tree) else []


class Metrics:
    def __init__(self, source: Forest, destination: Forest):
        self._source = source
        self._destination = destination
        self._cluster_cache = {}

    @cachedmethod(lambda self: self._cluster_cache)
    def _clusters(self, tau: float, metric: METRIC_FUNC) -> tuple[tuple[int, ...], tuple[int, ...]]:
        source_clustering = entity_labels(self._source, tau=tau, metric=metric)
        destination_clustering = entity_labels(self._destination, tau=tau, metric=None)

        entities = sorted(set(source_clustering.keys()) | set(destination_clustering.keys()))

        source_labels = tuple(source_clustering.get(ent, -i) for i, ent in enumerate(entities))
        destination_labels = tuple(destination_clustering.get(ent, -i) for i, ent in enumerate(entities))

        return source_labels, destination_labels

    def coverage(self) -> float:
        source_entities = {
            f"{subtree.label().name}${' '.join(subtree)}"
            for tree in self._source
            for subtree in tree.subtrees(lambda x: has_type(x, NodeType.ENT))
        }
        destination_entities = {
            f"{subtree.label().name}${' '.join(subtree)}"
            for tree in self._destination
            for subtree in tree.subtrees(lambda x: has_type(x, NodeType.ENT))
        }

        return jaccard(source_entities, destination_entities)

    def similarity(self, *, metric: METRIC_FUNC = DEFAULT_METRIC) -> float:
        """
        Calculates the similarity between the source and destination trees using the
        specified metric function and returns the average similarity score.

        Higher is better.

        :param metric: The similarity metric function used to compute the similarity between subtrees.
        :return: The average similarity score for all tree pairs in source and destination forests.
        """
        return sum(
            similarity(src_tree, dst_tree, metric=metric)
            for src_tree, dst_tree in zip(self._source, self._destination, strict=True)
        ) / len(self._source)

    def edit_distance(self) -> int:
        """
        Computes the total edit distance between corresponding source and destination trees.

        The method calculates the edit distance for each pair of source and destination trees using the APTED algorithm.
        The total edit distance is obtained by summing up the individual distances across all pairs of trees.

        Lower is better.

        :return: The total edit distance computed across all source and destination tree pairs.
        """
        return sum(
            APTED(src_tree, dst_tree, config=EditDistanceConfig()).compute_edit_distance()
            for src_tree, dst_tree in zip(self._source, self._destination, strict=True)
        )

    def cluster_ami(self, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> float:
        """
        Compute the Adjusted Mutual Information (AMI) score between source and destination clusters.
        The AMI score measures agreement while adjusting for random chance.

        Leverages the `adjusted_mutual_info_score` function from scikit-learn.

        Greater is better.

        :param tau: The similarity threshold for clustering.
        :param metric: The similarity metric function used to compute the similarity between subtrees.
        :return: The AMI score between the source and destination clusters.
        """
        from sklearn.metrics import adjusted_mutual_info_score

        source_labels, destination_labels = self._clusters(tau, metric)
        return adjusted_mutual_info_score(source_labels, destination_labels)

    def cluster_completeness(self, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> float:
        """
        Compute the completeness score between source and destination clusters.
        The AMI score measures agreement while adjusting for random chance.

        Leverages the `completeness_score` function from scikit-learn.

        Greater is better.

        :param tau: The similarity threshold for clustering.
        :param metric: The similarity metric function used to compute the similarity between subtrees.
        :return: The completeness score between the source and destination clusters.
        """
        from sklearn.metrics.cluster import completeness_score

        source_labels, destination_labels = self._clusters(tau, metric)
        return completeness_score(source_labels, destination_labels)

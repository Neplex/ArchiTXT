from collections.abc import Hashable
from itertools import combinations

import pandas as pd
from apted import APTED
from apted import Config as APTEDConfig
from cachetools import cachedmethod

from .schema import Schema
from .similarity import DEFAULT_METRIC, METRIC_FUNC, entity_labels, jaccard, similarity
from .tree import Forest, NodeType, Tree, has_type

__all__ = ['Metrics', 'confidence', 'dependency_score', 'redundancy_score']


def confidence(dataframe: pd.DataFrame, column: Hashable) -> float:
    """
    Computes the confidence score of the functional dependency ``X -> column`` in a DataFrame.

    The confidence score quantifies the strength of the association rule ``X -> column``,
    where ``X`` represents the set of all other attributes in the DataFrame.
    It is computed as the median of the confidence scores across all instantiated association rules.

    The confidence of each instantiated rule is calculated as the ratio of the consequent support
    (i.e., the count of each unique value in the specified column) to the antecedent support
    (i.e., the count of unique combinations of all other columns).
    A higher confidence score indicates a stronger dependency between the attributes.

    :param dataframe: A pandas DataFrame containing the data to analyze.
    :param column: The column for which to compute the confidence score.
    :return: The median confidence score or ``0.0`` if the data is empty.

    >>> data = pd.DataFrame({
    ...     'A': ['x', 'y', 'x', 'x', 'y'],
    ...     'B': [1, 2, 1, 3, 2]
    ... })
    >>> confidence(data, 'A')
    1.0
    >>> confidence(data, 'B')
    0.6666666666666666
    """
    consequent_support = dataframe.groupby(column).value_counts()
    antecedent_support = dataframe.drop(columns=[column]).value_counts()
    confidence_score = consequent_support / antecedent_support

    return confidence_score.median() if not consequent_support.empty else 0.0


def dependency_score(dataframe: pd.DataFrame, attributes: list[Hashable]) -> float:
    """
    Computes the dependency score of a subset of attributes in a DataFrame.

    The dependency score measures the strength of the functional dependency in the given subset of attributes.
    It is defined as the maximum confidence score among all attributes in the subset,
    treating each attribute as a potential consequent of a functional dependency.

    :param dataframe: A pandas DataFrame containing the data to analyze.
    :param attributes: A list of attributes to evaluate for functional dependencies.
    :return: The maximum confidence score among the given attributes.

    >>> data = pd.DataFrame({
    ...     'A': ['x', 'y', 'x', 'x', 'y'],
    ...     'B': [1, 2, 1, 3, 2]
    ... })
    >>> dependency_score(data, ['A', 'B'])
    1.0
    """
    return pd.Series(attributes).map(lambda x: confidence(dataframe[attributes], x)).max()


def redundancy_score(dataframe: pd.DataFrame, tau: float = 1.0) -> float:
    """
    Computes the redundancy score for an entire DataFrame.

    The overall redundancy score measures the fraction of rows that are redundant in at least one subset of attributes
    that satisfies a functional dependency above a given threshold tau.

    :param dataframe: A pandas DataFrame containing the data to analyze.
    :param tau: The dependency threshold to determine redundancy (default is 1.0).
    :return: The proportion of redundant rows in the dataset.

    >>> data = pd.DataFrame({
    ...     'A': ['x', 'y', 'x', 'x', 'y'],
    ...     'B': [1, 2, 1, 3, 2]
    ... })
    >>> dependency_score(data, ['A', 'B'])
    1.0
    """
    # Create a boolean Series initialized to False for all rows.
    duplicates = pd.Series(False, index=dataframe.index)
    attributes = dataframe.columns.tolist()

    # For each candidate attribute set, if its dependency score is above the threshold,
    # mark the rows that are duplicates on that set.
    for i in range(2, len(attributes)):
        for attrs in combinations(attributes, i):
            attrs = list(attrs)
            if dependency_score(dataframe, attrs) >= tau:
                duplicates |= dataframe[attrs].dropna().duplicated(keep=False)

    # The table-level redundancy is the fraction of rows that are duplicates in at least one candidate set.
    return duplicates.sum() / dataframe.shape[0]


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

    def redundancy(self, *, tau: float = 1.0):
        """
        Computes the redundancy score for the entire instance.

        The overall redundancy score measures the fraction of rows that are redundant in at least
        one subset of attributes that satisfies a functional dependency above a given threshold tau.

        :param tau: The dependency threshold to determine redundancy (default is 1.0).
        :return: The proportion of redundant rows in the dataset.
        """
        schema = Schema.from_forest(self._destination)
        datasets = schema.extract_datasets(self._destination)
        group_redundancy = pd.Series(datasets.values()).map(lambda df: redundancy_score(df, tau=tau))
        redundancy = group_redundancy[group_redundancy > 0].median()

        return redundancy if redundancy is not pd.NA else 0.0

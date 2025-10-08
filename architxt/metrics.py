from collections import Counter
from collections.abc import Collection, Hashable
from itertools import combinations
from operator import attrgetter
from typing import Any

import pandas as pd
from cachetools import cachedmethod

from .schema import Schema
from .similarity import DEFAULT_METRIC, METRIC_FUNC, entity_labels, jaccard
from .tree import Forest, NodeType, Tree, has_type

__all__ = ['Metrics', 'confidence', 'dependency_score', 'redundancy_score']


def confidence(dataframe: pd.DataFrame, column: str) -> float:
    """
    Compute the confidence score of the functional dependency ``X -> column`` in a DataFrame.

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


def dependency_score(dataframe: pd.DataFrame, attributes: Collection[str]) -> float:
    """
    Compute the dependency score of a subset of attributes in a DataFrame.

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
    return pd.Series(list(attributes)).map(lambda x: confidence(dataframe[list(attributes)], x)).max()


def redundancy_score(dataframe: pd.DataFrame, tau: float = 1.0) -> float:
    """
    Compute the redundancy score for an entire DataFrame.

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
            if dependency_score(dataframe, attrs) >= tau:
                duplicates |= dataframe[list(attrs)].dropna().duplicated(keep=False)

    # The table-level redundancy is the fraction of rows that are duplicates in at least one candidate set.
    return duplicates.sum() / dataframe.shape[0]


class Metrics:
    """
    A class to compute various comparison metrics between the original and modified forest states.

    This class is designed to track and measure changes in a forest structure that is modified in-place.
    It stores the initial state of the forest when instantiated and provides methods to compare
    the current state with the initial state using various metrics.

    :param forest: The forest to analyze
    :param tau: Threshold for subtree similarity when clustering.
    :param metric: The metric function used to compute similarity between subtrees.

    >>> forest = [tree1, tree2, tree3]  # doctest: +SKIP
    ... metrics = Metrics(forest, tau=0.7)
    ... # Modify forest in-place
    ... simplify(forest, tau=0.7)
    ... # Update the metrics object
    ... metrics.update()
    ... # Compare with the initial state
    ... similarity = metrics.cluster_ami()
    """

    def __init__(self, forest: Forest, *, tau: float, metric: METRIC_FUNC = DEFAULT_METRIC) -> None:
        self._cache: dict[Hashable, Any] = {}
        self._forest = forest
        self._tau = tau
        self._metric = metric

        self._source_schema = Schema.from_forest(self._forest)
        self._current_schema = self._source_schema

        self._datasets = self._current_schema.extract_datasets(self._forest)

        self._source_entities = {entity.oid.hex for tree in self._forest for entity in tree.entities()}
        self._current_entities = self._source_entities

        self._source_label_count = Counter(subtree.label for tree in self._forest for subtree in tree.subtrees())
        self._current_label_count = self._source_label_count

        self._source_clustering = entity_labels(self._forest, tau=self._tau, metric=self._metric)
        self._current_clustering = entity_labels(self._forest, tau=self._tau)

    def update(self, forest: Forest | None = None) -> None:
        """
        Update the internal state of the metrics object.

        :param forest: The forest to compare against, else read the original modified forest
        """
        self._cache.clear()

        forest = forest or self._forest

        self._current_schema = Schema.from_forest(forest)
        self._datasets = self._current_schema.extract_datasets(forest)
        self._current_entities = {entity.oid.hex for tree in forest for entity in tree.entities()}
        self._current_label_count = Counter(subtree.label for tree in forest for subtree in tree.subtrees())
        self._current_clustering = entity_labels(forest, tau=self._tau, metric=self._metric)

    @cachedmethod(attrgetter('_cache'))
    def _cluster_labels(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        entities = sorted(self._source_clustering.keys() | self._current_clustering.keys())

        # Use negative indices for entities that are not present
        source_labels = tuple(self._source_clustering.get(ent, str(-i)) for i, ent in enumerate(entities))
        current_labels = tuple(self._current_clustering.get(ent, str(-i)) for i, ent in enumerate(entities))

        return source_labels, current_labels

    @cachedmethod(attrgetter('_cache'))
    def coverage(self) -> float:
        """
        Compute the coverage between initial and current forest states.

        Coverage is measured using the :py:func:`~architxt.similarity.jaccard` similarity between the sets of entities
        in the original and current states.

            Greater is better.

        :return: Coverage score between 0 and 1, where 1 indicates identical entity sets
        """
        return jaccard(self._source_entities, self._current_entities)

    @cachedmethod(attrgetter('_cache'))
    def cluster_ami(self) -> float:
        """
        Compute the Adjusted Mutual Information (AMI) score between original and current clusters.

        The AMI score measures the agreement between two clustering while adjusting for chance.
        It uses :py:func:`sklearn.metrics.adjusted_mutual_info_score` under the hood.

            Greater is better.

        :return: Score between -1 and 1, where:
            - 1 indicates perfect agreement
            - 0 indicates random label assignments
            - negative values indicate worse than random labeling
        """
        from sklearn.metrics import adjusted_mutual_info_score

        source_labels, current_labels = self._cluster_labels()
        return adjusted_mutual_info_score(source_labels, current_labels)

    @cachedmethod(attrgetter('_cache'))
    def cluster_completeness(self) -> float:
        """
        Compute the completeness score between original and current clusters.

        Completeness measures if all members of a given class are assigned to the same cluster.
        It uses :py:func:`sklearn.metrics.completeness_score` under the hood.

            Greater is better.

        :return: Score between 0 and 1, where:
            - 1 indicates perfect completeness
            - 0 indicates worst possible completeness
        """
        from sklearn.metrics.cluster import completeness_score

        source_labels, current_labels = self._cluster_labels()
        return completeness_score(source_labels, current_labels)

    @cachedmethod(attrgetter('_cache'))
    def redundancy(self, *, tau: float = 1.0) -> float:
        """
        Compute the redundancy score for the current forest state.

        The overall redundancy score measures the fraction of rows that are redundant in at least
        one subset of attributes that satisfies a functional dependency above a given threshold tau.

            Lower is better.

        :param tau: The dependency threshold to determine redundancy (default is 1.0).
        :return: Score between 0 and 1, where:
            - 0 indicates no redundancy
            - 1 indicates complete redundancy
        """
        group_redundancy = pd.Series(self._datasets.values()).map(lambda df: redundancy_score(df, tau=tau))
        redundancy = group_redundancy[group_redundancy > 0].median()

        return redundancy if redundancy is not pd.NA else 0.0

    @cachedmethod(attrgetter('_cache'))
    def group_overlap_origin(self) -> float:
        """
        Get the origin schema group overlap ratio.

        See: :py:meth:`architxt.schema.Schema.group_overlap`
        """
        return self._source_schema.group_overlap

    @cachedmethod(attrgetter('_cache'))
    def group_overlap(self) -> float:
        """
        Get the schema group overlap ratio.

        See: :py:meth:`architxt.schema.Schema.group_overlap`
        """
        return self._current_schema.group_overlap

    @cachedmethod(attrgetter('_cache'))
    def group_balance_score_origin(self) -> float:
        """
        Get the origin group balance score.

        See: :py:meth:`architxt.schema.Schema.group_balance_score`
        """
        return self._source_schema.group_balance_score

    @cachedmethod(attrgetter('_cache'))
    def group_balance_score(self) -> float:
        """
        Get the group balance score.

        See: :py:meth:`architxt.schema.Schema.group_balance_score`
        """
        return self._current_schema.group_balance_score

    @cachedmethod(attrgetter('_cache'))
    def num_productions_origin(self) -> int:
        """Get the number of productions in the origin schema."""
        return len(self._source_schema.productions())

    @cachedmethod(attrgetter('_cache'))
    def num_productions(self) -> int:
        """Get the number of productions in the schema."""
        return len(self._current_schema.productions())

    @cachedmethod(attrgetter('_cache'))
    def ratio_productions(self) -> float:
        """Get the ratio of productions in the schema compare to the origin schema."""
        origin_productions = self.num_productions_origin()
        return self.num_productions() / origin_productions if origin_productions else 0

    @cachedmethod(attrgetter('_cache'))
    def num_non_terminal(self) -> int:
        """Get the number of non-terminal nodes in the schema."""
        return len(self._current_label_count)

    @cachedmethod(attrgetter('_cache'))
    def num_nodes(self) -> int:
        """Get the total number of nodes in the forest."""
        return sum(self._current_label_count.values())

    @cachedmethod(attrgetter('_cache'))
    def num_unlabeled_nodes(self) -> int:
        """Get the total number of unlabeled nodes in the forest."""
        return sum(count for label, count in self._current_label_count.items() if not has_type(label))

    @cachedmethod(attrgetter('_cache'))
    def ratio_unlabeled_nodes(self) -> float:
        """Get the ratio of unlabeled nodes in the forest."""
        nb_nodes = self.num_nodes()
        return self.num_unlabeled_nodes() / nb_nodes if nb_nodes else 0

    @cachedmethod(attrgetter('_cache'))
    def num_distinct_type(self, node_type: NodeType) -> int:
        """
        Get the number of distinct labels in the schema that match the given node type.

        :param node_type: The type to filter by.
        """
        return sum(has_type(label, node_type) for label in self._current_label_count)

    @cachedmethod(attrgetter('_cache'))
    def num_type(self, node_type: NodeType) -> int:
        """
        Get the total number of nodes in the forest that match the given node type.

        :param node_type: The type to filter by.
        """
        return sum(count for label, count in self._current_label_count.items() if has_type(label, node_type))

    @cachedmethod(attrgetter('_cache'))
    def ratio_type(self, node_type: NodeType) -> float:
        """
        Return the average number of nodes per distinct label for the given node type.

        :param node_type: The type to filter by.
        """
        nb_collections = self.num_distinct_type(node_type)
        return self.num_type(node_type) / nb_collections if nb_collections else 0

    def log_to_mlflow(self, iteration: int, *, debug: bool = False) -> None:
        """
        Log various metrics related to a forest of trees and equivalent subtrees.

        :param iteration: The current iteration number for logging.
        :param debug: Whether to enable debug logging.
        """
        import mlflow

        if not mlflow.active_run():
            return

        # Log the calculated metrics
        mlflow.log_metrics(
            {
                'nodes.count': self.num_nodes(),
                'unlabeled.count': self.num_unlabeled_nodes(),
                'unlabeled.ratio': self.ratio_unlabeled_nodes(),
                'redundancy': self.redundancy(),
                # Clustering
                'clustering.cluster_count': len(set(self._current_clustering.values())),
                'clustering.ami': self.cluster_ami(),
                'clustering.completeness': self.cluster_completeness(),
                # Entities
                'entities.coverage': self.coverage(),
                'entities.count': self.num_type(NodeType.ENT),
                'entities.distinct_count': self.num_distinct_type(NodeType.ENT),
                'entities.ratio': self.ratio_type(NodeType.ENT),
                # Groups
                'groups.count': self.num_type(NodeType.GROUP),
                'groups.distinct_count': self.num_distinct_type(NodeType.GROUP),
                'groups.ratio': self.ratio_type(NodeType.GROUP),
                # Relations
                'relations.count': self.num_type(NodeType.REL),
                'relations.distinct_count': self.num_distinct_type(NodeType.REL),
                'relations.ratio': self.ratio_type(NodeType.REL),
                # Collections
                'collections.count': self.num_type(NodeType.COLL),
                'collections.distinct_count': self.num_distinct_type(NodeType.COLL),
                'collections.ratio': self.ratio_type(NodeType.COLL),
                # Schema
                'schema.overlap': self.group_overlap(),
                'schema.balance': self.group_balance_score(),
                'schema.productions': self.num_productions(),
                'schema.non_terminal': self.num_non_terminal(),
            },
            step=iteration,
        )

        if debug:
            rooted_forest = Tree('ROOT', (tree.copy() for tree in self._forest))
            mlflow.log_text(rooted_forest.to_svg(), f'debug/{iteration}/tree.html')
            mlflow.log_text(self._current_schema.as_cfg(), f'debug/{iteration}/schema.txt')

            cluster_table = pd.DataFrame(
                self._current_clustering.items(),
                columns=['tree oid', 'cluster'],
                dtype=str,
            ).sort_values('cluster')
            mlflow.log_table(cluster_table, f'debug/{iteration}/clusters.json')

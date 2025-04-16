from collections import Counter
from collections.abc import Collection

import mlflow

from architxt.metrics import Metrics
from architxt.schema import Schema
from architxt.similarity import METRIC_FUNC, TREE_CLUSTER
from architxt.tree import Forest, NodeType, Tree, has_type

__all__ = [
    'distribute_evenly',
    'log_clusters',
    'log_instance_comparison_metrics',
    'log_metrics',
    'log_schema',
]


def distribute_evenly(trees: Collection[Tree], n: int) -> list[list[Tree]]:
    """
    Distribute a collection of trees into `n` sub-collections with approximately equal total complexity.

    Complexity is determined by the number of leaves in each tree.
    The function attempts to create `n` chunks, but if there are fewer elements than `n`,
    it will create one chunk per element.

    :param trees: A collection of trees.
    :param n: The number of sub-collections to create.
    :return: A list of `n` sub-collections, with trees distributed to balance complexity.
    :raises ValueError: If `n` is less than 1.
    """
    if n < 1:
        msg = "The number of sub-collections 'n' must be at least 1."
        raise ValueError(msg)

    n = min(n, len(trees))

    # Sort trees in descending order of their leaf count for a greedy allocation.
    sorted_trees = sorted(trees, key=lambda tree: len(tree.leaves()), reverse=True)

    chunks: list[list[Tree]] = [[] for _ in range(n)]
    chunk_complexities = [0] * n

    # Greedy distribution: Assign each tree to the chunk with the smallest current complexity.
    for tree in sorted_trees:
        least_complex_chunk_index = chunk_complexities.index(min(chunk_complexities))
        chunks[least_complex_chunk_index].append(tree)
        chunk_complexities[least_complex_chunk_index] += len(tree.leaves())

    return chunks


def log_instance_comparison_metrics(
    iteration: int, old_forest: Forest, new_forest: Forest, tau: float, metric: METRIC_FUNC
) -> None:
    """
    Log comparison metrics to see the evolution of the rewriting for a specific iteration.

    :param iteration: The current iteration number for logging.
    :param old_forest: The initial forest to compare against.
    :param new_forest: The updated forest to compare with.
    :param tau: The similarity threshold for clustering.
    :param metric: The similarity metric function used to compute the similarity between subtrees.
    """
    if not mlflow.active_run():
        return

    metrics = Metrics(old_forest, new_forest)
    mlflow.log_metrics(
        {
            'coverage': metrics.coverage(),
            'similarity': metrics.similarity(),
            'edit_distance': metrics.edit_distance(),
            'cluster_ami': metrics.cluster_ami(tau=tau, metric=metric),
            'cluster_completeness': metrics.cluster_completeness(tau=tau, metric=metric),
        },
        step=iteration,
    )


def log_metrics(iteration: int, forest: Forest, equiv_subtrees: TREE_CLUSTER | None = None) -> None:
    """
    Log various metrics related to a forest of trees and equivalent subtrees.

    This function calculates and logs the metrics that provide insights into the forest's structure, including counts of
    production rules, labeled and unlabeled nodes, and entity/group/collection/relation statistics.

    :param iteration: The current iteration number for logging.
    :param forest: A forest of tree objects to analyze.
    :param equiv_subtrees: A set of clusters representing equivalent subtrees.
    :return: None
    """
    if not mlflow.active_run():
        return

    # Count labels for all nodes in the forest
    label_counts = Counter(subtree.label for tree in forest for subtree in tree.subtrees())

    # Calculate the number of unlabeled nodes
    num_unlabeled = sum(not has_type(label) for label in label_counts)
    unlabeled_ratio = num_unlabeled / len(label_counts) if len(label_counts) else 0

    # Entity statistics
    num_entities = sum(has_type(label, NodeType.ENT) for label in label_counts)
    num_entity_instances = sum(label_counts[label] for label in label_counts if has_type(label, NodeType.ENT))
    entity_ratio = num_entity_instances / num_entities if num_entities else 0

    # Group statistics
    num_groups = sum(has_type(label, NodeType.GROUP) for label in label_counts)
    num_group_instances = sum(label_counts[label] for label in label_counts if has_type(label, NodeType.GROUP))
    group_ratio = num_group_instances / num_groups if num_groups else 0

    # Relation statistics
    num_relations = sum(has_type(label, NodeType.REL) for label in label_counts)
    num_relation_instances = sum(label_counts[label] for label in label_counts if has_type(label, NodeType.REL))
    relation_ratio = num_relation_instances / num_relations if num_relations else 0

    # Collection statistics
    num_collections = sum(has_type(label, NodeType.COLL) for label in label_counts)
    num_collection_instances = sum(label_counts[label] for label in label_counts if has_type(label, NodeType.COLL))
    collection_ratio = num_collection_instances / num_collections if num_collections else 0

    # Log the calculated metrics
    mlflow.log_metrics(
        {
            'non_terminal_nodes': len(label_counts),
            'unlabeled_nodes': num_unlabeled,
            'unlabeled_nodes_ratio': unlabeled_ratio,
            'equiv_subtrees': len(equiv_subtrees) if equiv_subtrees else 0,
            'entity_type_total': num_entities,
            'entity_instance_total': num_entity_instances,
            'entity_instance_ratio': entity_ratio,
            'group_type_total': num_groups,
            'group_instance_total': num_group_instances,
            'group_instance_ratio': group_ratio,
            'relation_type_total': num_relations,
            'relation_instance_total': num_relation_instances,
            'relation_instance_ratio': relation_ratio,
            'collection_type_total': num_collections,
            'collection_instance_total': num_collection_instances,
            'collection_instance_ratio': collection_ratio,
        },
        step=iteration,
    )


def log_clusters(iteration: int, equiv_subtrees: TREE_CLUSTER) -> None:
    """
    Log information about the clusters of equivalent subtrees.

    This function processes each cluster of subtrees, extracting the entity labels, count, and maximum label length,
    and then logs this information using MLFlow.

    :param iteration: The current iteration number.
    :param equiv_subtrees: The set of equivalent subtrees to process.
    """
    if not mlflow.active_run():
        return

    elems = []
    count = []
    max_len = []

    for equiv_class in equiv_subtrees:
        elems.append({str(equiv_tree.label) for equiv_tree in equiv_class})
        count.append(len(equiv_class))
        max_len.append(len(max(equiv_class, key=len)))

    mlflow.log_table(
        {
            'elems': elems,
            'count': count,
            'max_len': max_len,
        },
        f'debug/{iteration}/tree.json',
    )


def log_schema(iteration: int, forest: Forest) -> None:
    """
    Log the schema to MLFlow.

    :param iteration: The current iteration number for logging.
    :param forest: A forest of tree objects to analyze.
    """
    if not mlflow.active_run():
        return

    schema = Schema.from_forest(forest, keep_unlabelled=True)

    mlflow.log_metrics(
        {
            'num_productions': len(schema.productions()),
            'overlap': schema.group_overlap,
            'balance': schema.group_balance_score,
        },
        step=iteration,
    )
    mlflow.log_text(schema.as_cfg(), f'debug/{iteration}/schema.txt')

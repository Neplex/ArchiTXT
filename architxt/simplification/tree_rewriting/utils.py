from collections import Counter
from collections.abc import Collection

import mlflow

from architxt.model import NodeType
from architxt.schema import Schema
from architxt.similarity import TREE_CLUSTER
from architxt.tree import Forest, Tree, has_type

__all__ = [
    'distribute_evenly',
    'log_clusters',
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
        raise ValueError("The number of sub-collections 'n' must be at least 1.")

    n = min(n, len(trees))

    # Sort trees in descending order of their leaf count for a greedy allocation.
    sorted_trees = sorted(trees, key=lambda tree: len(tree.leaves()), reverse=True)

    chunks = [[] for _ in range(n)]
    chunk_complexities = [0] * n

    # Greedy distribution: Assign each tree to the chunk with the smallest current complexity.
    for tree in sorted_trees:
        least_complex_chunk_index = chunk_complexities.index(min(chunk_complexities))
        chunks[least_complex_chunk_index].append(tree)
        chunk_complexities[least_complex_chunk_index] += len(tree.leaves())

    return chunks


def log_metrics(iteration: int, forest: Forest, equiv_subtrees: TREE_CLUSTER = ()) -> None:
    """
    Logs various metrics related to a forest of trees and equivalent subtrees.

    This function calculates and logs the metrics that provide insights into the forest's structure, including counts of
    production rules, labeled and unlabeled nodes, and entity/group/collection/relation statistics.

    :param iteration: The current iteration number for logging.
    :param forest: A forest of tree objects to analyze.
    :param equiv_subtrees: A set of clusters representing equivalent subtrees.
    :return: None
    """
    # Count labels for all nodes in the forest
    label_counts = Counter(str(subtree.label()) for tree in forest for subtree in tree.subtrees())

    # Calculate the number of unlabeled nodes
    num_unlabeled = sum(not has_type(label) for label in label_counts)

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
            'num_non_terminal_nodes': len(label_counts),
            'num_unlabeled_nodes': num_unlabeled,
            'num_equiv_subtrees': len(equiv_subtrees),
            'num_entities': num_entities,
            'num_entity_instances': num_entity_instances,
            'entity_ratio': entity_ratio,
            'num_groups': num_groups,
            'num_group_instances': num_group_instances,
            'group_ratio': group_ratio,
            'num_relations': num_relations,
            'num_relation_instances': num_relation_instances,
            'relation_ratio': relation_ratio,
            'num_collections': num_collections,
            'num_collection_instances': num_collection_instances,
            'collection_ratio': collection_ratio,
        },
        step=iteration,
    )


def log_clusters(iteration: int, equiv_subtrees: TREE_CLUSTER) -> None:
    """
    Logs information about the clusters of equivalent subtrees.

    This function processes each cluster of subtrees, extracting the entity labels, count, and maximum label length,
    and then logs this information using MLFlow.

    :param iteration: The current iteration number.
    :param equiv_subtrees: The set of equivalent subtrees to process.
    """
    elems = []
    count = []
    max_len = []

    for equiv_class in equiv_subtrees:
        elems.append({str(equiv_tree.label()) for equiv_tree in equiv_class})
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
    schema = Schema.from_forest(forest)

    mlflow.log_metric('num_productions', len(schema.productions()))
    mlflow.log_text(schema.as_cfg(), f'debug/{iteration}/schema.txt')

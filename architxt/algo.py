import functools
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from multiprocessing import cpu_count

import mlflow
from nltk import TreePrettyPrinter
from tqdm import trange
from tqdm.contrib.concurrent import process_map

from architxt import operations
from architxt.db import Schema
from architxt.model import NodeType
from architxt.operations import OPERATION
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC, TREE_CLUSTER, equiv_cluster
from architxt.tree import Forest, Tree, has_type, reduce_all

DEFAULT_OPERATIONS = (
    operations.find_subgroups,
    operations.merge_groups,
    operations.find_collections,
    operations.find_relations,
    operations.find_collections,
    operations.reduce_bottom,
    operations.reduce_top,
)


def rewrite(
    forest: Forest,
    *,
    tau: float = 0.7,
    epoch: int = 100,
    min_support: int | None = None,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    edit_ops: Sequence[operations.OPERATION] = DEFAULT_OPERATIONS,
    debug: bool = False,
    max_workers: int | None = None,
) -> Forest:
    """
    Iteratively rewrites a forest by applying edit operations.

    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param epoch: Maximum number of rewriting steps.
    :param min_support: Minimum support of groups.
    :param metric:  The metric function used to compute similarity between subtrees.
    :param edit_ops: The list of operations to perform on the forest.
    :param debug: Whether to enable debug logging.
    :param max_workers: Number of parallel worker processes to use.

    :return: The rewritten forest.
    """
    min_support = min_support or max((len(forest) // 10), 2)
    max_workers = max_workers or cpu_count()

    mlflow.log_params(
        {
            'nb_sentences': len(forest),
            'tau': tau,
            'epoch': epoch,
            'min_support': min_support,
            'metric': metric.__name__,
            'edit_ops': ', '.join(f"{op_id}: {edit_op.__name__}" for op_id, edit_op in enumerate(edit_ops)),
        }
    )

    with mlflow.start_span('rewriting'):
        for iteration in trange(epoch, desc='rewrite trees'):
            with mlflow.start_span('iteration', attributes={'step': iteration}):
                forest, has_simplified = _rewrite_step(
                    iteration,
                    forest,
                    tau=tau,
                    min_support=min_support,
                    metric=metric,
                    edit_ops=edit_ops,
                    debug=debug,
                    max_workers=max_workers,
                )

                # Stop if no further simplifications are made
                if iteration > 0 and not has_simplified:
                    break

    return forest


def _rewrite_step(
    iteration: int,
    forest: Forest,
    *,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    edit_ops: Sequence[operations.OPERATION],
    debug: bool,
    max_workers: int,
) -> tuple[Forest, bool]:
    """
    Perform a single rewrite step on the forest.

    :param iteration: The current iteration number.
    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric:  The metric function used to compute similarity between subtrees.
    :param edit_ops: The list of operations to perform on the forest.
    :param debug: Whether to enable debug logging.
    :param max_workers: Number of parallel worker processes to use.

    :return: The updated forest and a flag indicating if simplifications occurred.
    """
    if debug:
        # Log the forest as SVG
        rooted_forest = Tree('ROOT', deepcopy(forest))
        mlflow.log_text(TreePrettyPrinter(rooted_forest).svg(), f'debug/{iteration}/tree.html')

    with mlflow.start_span('reduce_all'):
        for tree in forest:
            reduce_all(tree, {NodeType.ENT})

    with mlflow.start_span('equiv_cluster'):
        equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric)
        if debug:
            _log_clusters(iteration, equiv_subtrees)

    with mlflow.start_span('find_groups'):
        operations.find_groups(
            Tree('', []),
            equiv_subtrees=equiv_subtrees,
            tau=tau,
            min_support=min_support,
            metric=metric,
        )

    has_simplified = False
    for op_id, edit_op in enumerate(edit_ops):
        forest, simplified = _apply_operation(
            edit_op,
            forest,
            equiv_subtrees=equiv_subtrees,
            tau=tau,
            min_support=min_support,
            metric=metric,
            max_workers=max_workers,
        )

        if has_simplified := any(simplified):
            mlflow.log_metric('edit_op', op_id, step=iteration)
            break

    forest = _post_process(forest, tau, metric, max_workers=max_workers)

    _log_schema(iteration, forest)
    _log_metrics(iteration, forest, equiv_subtrees)

    return forest, has_simplified


def _post_process(
    forest: Forest,
    tau: float,
    metric: METRIC_FUNC,
    max_workers: int,
) -> Forest:
    """
    Post-process the forest to find and name relations and collections.

    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param metric: The metric function used to compute similarity between subtrees.
    :param max_workers: Number of parallel worker processes to use.

    :returns: The processed forest with named relations and collections.
    """
    equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric)

    forest, _ = _apply_operation(
        operations.find_relations,
        forest,
        equiv_subtrees=equiv_subtrees,
        tau=tau,
        min_support=0,
        metric=metric,
        naming_only=True,
        max_workers=max_workers,
        desc='[post-process] name_relations',
    )
    forest, _ = _apply_operation(
        operations.find_collections,
        forest,
        equiv_subtrees=equiv_subtrees,
        tau=tau,
        min_support=0,
        metric=metric,
        naming_only=True,
        max_workers=max_workers,
        desc='[post-process] name_collections',
    )

    return forest


def _apply_operation(
    operation: OPERATION,
    forest: Forest,
    *,
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    max_workers: int,
    desc: str = '',
    **kwargs,
) -> tuple[Forest, tuple[bool]]:
    """
    Apply the given operation to the forest.

    :param operation: The rewriting operation to apply.
    :param forest: The forest to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param max_workers: Number of parallel worker processes to use.
    :param desc: The description for tqdm progress bar, default to operation name.
    :return: The rewritten forest and the tuple of flags for each tree indicating if simplifications occurred.
    """
    desc = desc or operation.__name__
    chunk_size = max(10, len(forest) // (max_workers * 6))

    operation_fn = functools.partial(
        operation,
        equiv_subtrees=equiv_subtrees,
        tau=tau,
        min_support=min_support,
        metric=metric,
        **kwargs,
    )

    with mlflow.start_span(desc):
        result = process_map(
            operation_fn,
            forest,
            max_workers=max_workers,
            chunksize=chunk_size,
            leave=False,
            desc=desc,
        )

    return zip(*result)


def _log_metrics(iteration: int, forest: Forest, equiv_subtrees: TREE_CLUSTER = ()):
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
    label_counts = Counter(subtree.label() for tree in forest for subtree in tree.subtrees())

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


def _log_clusters(iteration: int, equiv_subtrees: TREE_CLUSTER):
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


def _log_schema(iteration: int, forest: Forest):
    """
    Log the schema to MLFlow.
    :param iteration: The current iteration number for logging.
    :param forest: A forest of tree objects to analyze.
    """
    schema = Schema.from_forest(forest)

    mlflow.log_metric('num_productions', len(schema.productions()))
    mlflow.log_text(schema.as_cfg(), f'debug/{iteration}/schema.txt')

import ctypes
import functools
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import Barrier, Manager, Value, cpu_count
from typing import cast

import mlflow
import more_itertools
from nltk import TreePrettyPrinter
from tqdm import tqdm, trange

from architxt import operations
from architxt.db import Schema
from architxt.model import NodeType
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

    forest, op_id = _apply_operations(
        edit_ops,
        forest,
        equiv_subtrees=equiv_subtrees,
        tau=tau,
        min_support=min_support,
        metric=metric,
        max_workers=max_workers,
    )

    if op_id is not None:
        mlflow.log_metric('edit_op', op_id, step=iteration)

    forest = _post_process(forest, tau, metric, max_workers=max_workers)

    _log_schema(iteration, forest)
    _log_metrics(iteration, forest, equiv_subtrees)

    return forest, op_id is not None


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

    forest, _ = _apply_operations(
        [
            ('[post-process] name_relations', functools.partial(operations.find_relations, naming_only=True)),
            ('[post-process] name_collections', functools.partial(operations.find_collections, naming_only=True)),
        ],
        forest,
        equiv_subtrees=equiv_subtrees,
        tau=tau,
        min_support=0,
        metric=metric,
        max_workers=max_workers,
        early_exit=False,
    )

    return forest


def _apply_operations(
    edit_ops: Sequence[operations.OPERATION] | Sequence[tuple[str, operations.OPERATION]],
    forest: Forest,
    *,
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    max_workers: int,
    early_exit: bool = True,
) -> tuple[Forest, int | None]:
    """
    Apply the given operations to the forest.
    The operations are applied in order, and we stop if one operation simplify at least one tree in the forest.

    :param edit_ops: The list of operations (or tuple name-operation) to perform on the forest.
    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param max_workers: Number of parallel worker processes to use.

    :return: The updated forest and the id of the operation that succeed.
    """
    max_workers = min(len(forest) // 100, max_workers)  # Cannot have less than 100 trees

    if not edit_ops:
        return forest, None

    if not isinstance(edit_ops[0], tuple):
        edit_ops = [(op.__name__, op) for op in edit_ops]

    with ProcessPoolExecutor(max_workers=max_workers) as executor, Manager() as manager:
        shared_equiv = manager.Value(ctypes.py_object, equiv_subtrees)
        simplification_operation = manager.Value(ctypes.c_int, -1)
        barrier = manager.Barrier(max_workers)

        futures = [
            executor.submit(
                _apply_operations_worker,
                idx,
                edit_ops,
                tuple(chunk),
                shared_equiv,
                tau,
                min_support,
                metric,
                early_exit,
                simplification_operation,
                barrier,
            )
            for idx, chunk in enumerate(more_itertools.divide(max_workers, forest))
            if chunk
        ]

        # Flatten the results and extract the simplification operation ID if any
        forest = [tree for chunk in as_completed(futures) for tree in chunk.result()]
        op_id = None if simplification_operation.value == -1 else cast(int, simplification_operation.value)

    return forest, op_id


def _apply_operations_worker(
    idx: int,
    edit_ops: Sequence[tuple[str, operations.OPERATION]],
    forest: Forest,
    equiv_subtrees: Value,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    early_exit: bool,
    simplification_operation: Value,
    barrier: Barrier,
) -> Forest:
    """
    Apply the given operation to the forest.

    :param edit_ops: The list of operations to perform on the forest.
    :param forest: The forest to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :return: The rewritten forest and the tuple of flags for each tree indicating if simplifications occurred.
    """
    for op_id, (op_name, op_fn) in enumerate(edit_ops):
        operation_fn = functools.partial(
            op_fn,
            equiv_subtrees=equiv_subtrees.value,
            tau=tau,
            min_support=min_support,
            metric=metric,
        )

        with mlflow.start_span(op_name):
            forest, simplified = zip(
                *map(operation_fn, tqdm(forest, desc=op_name, leave=False, position=idx + 1)), strict=False
            )

            if any(simplified):
                simplification_operation.value = op_id

            barrier.wait()  # Wait for all workers to finish this operation

            # If simplification has occurred in any worker, stop processing further operations.
            if early_exit and simplification_operation.value != -1:
                break

    return forest


def _log_metrics(iteration: int, forest: Forest, equiv_subtrees: TREE_CLUSTER = ()) -> None:
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


def _log_clusters(iteration: int, equiv_subtrees: TREE_CLUSTER) -> None:
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


def _log_schema(iteration: int, forest: Forest) -> None:
    """
    Log the schema to MLFlow.
    :param iteration: The current iteration number for logging.
    :param forest: A forest of tree objects to analyze.
    """
    schema = Schema.from_forest(forest)

    mlflow.log_metric('num_productions', len(schema.productions()))
    mlflow.log_text(schema.as_cfg(), f'debug/{iteration}/schema.txt')

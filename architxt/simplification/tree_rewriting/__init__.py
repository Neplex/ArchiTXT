import ctypes
import functools
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from copy import deepcopy
from multiprocessing import Manager, cpu_count
from multiprocessing.managers import ValueProxy
from multiprocessing.synchronize import Barrier

import mlflow
from mlflow.entities import SpanEvent
from nltk import TreePrettyPrinter
from tqdm.auto import tqdm, trange

from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC, TREE_CLUSTER, equiv_cluster
from architxt.tree import Forest, NodeLabel, NodeType, Tree, has_type, reduce_all

from .operations import (
    FindCollectionsOperation,
    FindRelationsOperation,
    FindSubGroupsOperation,
    MergeGroupsOperation,
    Operation,
    ReduceBottomOperation,
    ReduceTopOperation,
)
from .utils import distribute_evenly, log_clusters, log_instance_comparison_metrics, log_metrics, log_schema

__all__ = ['apply_operations', 'create_group', 'find_groups', 'rewrite']

DEFAULT_OPERATIONS: Sequence[type[Operation]] = (
    FindSubGroupsOperation,
    MergeGroupsOperation,
    FindCollectionsOperation,
    FindRelationsOperation,
    FindCollectionsOperation,
    ReduceBottomOperation,
    ReduceTopOperation,
)


def rewrite(
    forest: Forest,
    *,
    tau: float = 0.7,
    epoch: int = 100,
    min_support: int | None = None,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    edit_ops: Sequence[type[Operation]] = DEFAULT_OPERATIONS,
    debug: bool = False,
    max_workers: int | None = None,
) -> Forest:
    """
    Rewrite a forest by applying edit operations iteratively.

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
    if not forest:
        return forest

    min_support = min_support or max((len(forest) // 10), 2)
    max_workers = min(len(forest) // 100, max_workers or cpu_count()) or 1  # Cannot have less than 100 trees

    if mlflow.active_run():
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

        equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric)
        log_metrics(0, forest, equiv_subtrees)
        log_schema(0, forest)
        log_clusters(0, equiv_subtrees)
        log_instance_comparison_metrics(0, forest, forest, tau, metric)

        if debug:
            # Log the forest as SVG
            rooted_forest = Tree('ROOT', deepcopy(forest))
            mlflow.log_text(TreePrettyPrinter(rooted_forest).svg(), 'debug/0/tree.html')

    new_forest = deepcopy(forest)
    mp_ctx = multiprocessing.get_context('spawn')

    with mlflow.start_span('rewriting'), ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        for iteration in trange(1, epoch, desc='rewrite trees'):
            with mlflow.start_span('iteration', attributes={'step': iteration}):
                new_forest, has_simplified = _rewrite_step(
                    iteration,
                    new_forest,
                    tau=tau,
                    min_support=min_support,
                    metric=metric,
                    edit_ops=edit_ops,
                    debug=debug,
                    executor=executor,
                )

                log_instance_comparison_metrics(iteration, forest, new_forest, tau, metric)

                # Stop if no further simplifications are made
                if iteration > 0 and not has_simplified:
                    break

        new_forest, _ = _post_process(new_forest, tau=tau, metric=metric, executor=executor)
        log_instance_comparison_metrics(iteration + 1, forest, new_forest, tau, metric)

    return new_forest


def _rewrite_step(
    iteration: int,
    forest: Forest,
    *,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    edit_ops: Sequence[type[Operation]],
    debug: bool,
    executor: ProcessPoolExecutor,
) -> tuple[Forest, bool]:
    """
    Perform a single rewrite step on the forest.

    :param iteration: The current iteration number.
    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param edit_ops: The list of operations to perform on the forest.
    :param debug: Whether to enable debug logging.
    :param executor: A pool executor to parallelize the processing of the forest.

    :return: The updated forest and a flag indicating if simplifications occurred.
    """
    if mlflow.active_run() and debug:
        # Log the forest as SVG
        rooted_forest = Tree('ROOT', deepcopy(forest))
        mlflow.log_text(TreePrettyPrinter(rooted_forest).svg(), f'debug/{iteration}/tree.html')

    with mlflow.start_span('reduce_all'):
        for tree in forest:
            reduce_all(tree, {NodeType.ENT})

    with mlflow.start_span('equiv_cluster'):
        equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric, _step=iteration if debug else None)
        if debug:
            log_clusters(iteration, equiv_subtrees)

    with mlflow.start_span('find_groups'):
        find_groups(equiv_subtrees, min_support)

    forest, op_id = apply_operations(
        [operation(tau=tau, min_support=min_support, metric=metric) for operation in edit_ops],
        forest,
        equiv_subtrees=equiv_subtrees,
        executor=executor,
    )

    if op_id is not None:
        mlflow.log_metric('edit_op', op_id, step=iteration)

    if mlflow.active_run():
        renamed_forest, equiv_subtrees = _post_process(forest, tau=tau, metric=metric, executor=executor)
        log_schema(iteration, renamed_forest)
        log_metrics(iteration, renamed_forest, equiv_subtrees)

    return forest, op_id is not None


def _post_process(
    forest: Forest,
    *,
    tau: float,
    metric: METRIC_FUNC,
    executor: ProcessPoolExecutor,
) -> tuple[Forest, TREE_CLUSTER]:
    """
    Post-process the forest to find and name relations and collections.

    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param metric: The metric function used to compute similarity between subtrees.
    :param executor: A pool executor to parallelize the processing of the forest.

    :returns: The processed forest with named relations and collections.
    """
    equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric)

    forest, _ = apply_operations(
        [
            (
                '[post-process] name_relations',
                FindRelationsOperation(tau=tau, min_support=0, metric=metric, naming_only=True),
            ),
            (
                '[post-process] name_collections',
                FindCollectionsOperation(tau=tau, min_support=0, metric=metric, naming_only=True),
            ),
        ],
        forest,
        equiv_subtrees=equiv_subtrees,
        early_exit=False,
        executor=executor,
    )

    return forest, equiv_subtrees


def apply_operations(
    edit_ops: Sequence[Operation | tuple[str, Operation]],
    forest: Forest,
    *,
    equiv_subtrees: TREE_CLUSTER,
    early_exit: bool = True,
    executor: ProcessPoolExecutor,
) -> tuple[Forest, int | None]:
    """
    Apply a sequence of edit operations to a forest, potentially simplifying its structure.

    Each operation in `edit_ops` is applied to the forest in the provided order.
    If `early_exit` is enabled, the function stops as soon as an operation successfully simplifies at least one tree.
    Otherwise, all operations are applied.

    :param edit_ops: A sequence of operations to apply to the forest.
                     Each operation can either be a callable or a tuple `(name, callable)`
                     where `name` is a string identifier for the operation.
    :param forest: The input forest (a collection of trees) on which operations are applied.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param early_exit: A boolean flag indicating whether to stop after the first successful operation.
                       If `False`, all operations are applied.
    :param executor: A pool executor to parallelize the processing of the forest.

    :return: A tuple composed of:
        - The updated forest after applying the operations.
        - The index of the operation that successfully simplified a tree, or `None` if no operation succeeded.
    """
    if not edit_ops:
        return forest, None

    edit_ops_names = [(op.name, op) if isinstance(op, Operation) else op for op in edit_ops]
    chunks = distribute_evenly(forest, executor._max_workers)

    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    with Manager() as manager:
        shared_equiv = manager.Value(ctypes.py_object, equiv_subtrees, lock=False)
        simplification_operation = manager.Value(ctypes.c_int, -1, lock=False)
        barrier = manager.Barrier(len(chunks))

        futures = [
            executor.submit(
                _apply_operations_worker,
                idx,
                edit_ops_names,
                tuple(chunk),
                shared_equiv,
                early_exit,
                simplification_operation,
                barrier,
                run_id,
            )
            for idx, chunk in enumerate(chunks)
        ]

        forest = []
        for future in as_completed(futures):
            trees, request_id = future.result()
            forest.extend(trees)

            worker_trace = mlflow.get_trace(request_id)
            mlflow.add_trace(worker_trace)

        op_id = simplification_operation.get()

    return forest, op_id if op_id >= 0 else None


def _apply_operations_worker(
    idx: int,
    edit_ops: Sequence[tuple[str, Operation]],
    forest: Forest,
    shared_equiv_subtrees: ValueProxy[set[tuple[Tree, ...]]],
    early_exit: bool,
    simplification_operation: ValueProxy[int],
    barrier: Barrier,
    run_id: str | None,
) -> tuple[Forest, str]:
    """
    Apply the given operation to the forest.

    MLflow's tracing buffers spans in memory and exports the entire trace only when the root span concludes.
    This design does not inherently support multiprocessing, as spans created in separate processes are isolated
    and cannot be automatically aggregated into a single trace.
    As a result, we need to manually export spans from each subprocess and send them to the main process.
    To achieve this, an independent trace is created for each worker, which is then merged with the main trace.
    This means we SHOULD NOT pass OpenTelemetry context to the subprocess.
    Only the request id is sent, and we let the main process retrieve the trace from the tracking store
    See :py:func:`mlflow.tracing.fluent.add_trace`

    :param idx: The index of the worker.
    :param edit_ops: The list of operations to perform on the forest.
    :param forest: The forest to perform on.
    :param shared_equiv_subtrees: The shared set of equivalent subtrees.
    :param early_exit: A boolean flag indicating whether to stop after the first successful operation.
                       If `False`, all operations are applied.
    :param simplification_operation: A shared integer value to store the index of the operation that simplified a tree.
    :param barrier: A barrier to synchronize the workers before starting the next operation.
    :param run_id: The Mlflow run_id to link to.
    :return: The rewritten forest and the execution trace.
    """
    equiv_subtrees = shared_equiv_subtrees.get()

    with (
        mlflow.start_run(run_id=run_id) if run_id else nullcontext(),
        mlflow.start_span(
            "worker",
            attributes={
                'worker_id': idx,
                'batch': len(forest),
            },
        ) as span,
    ):
        request_id = span.request_id

        for op_id, (name, operation) in enumerate(edit_ops):
            op_fn = functools.partial(operation.apply, equiv_subtrees=equiv_subtrees)

            with mlflow.start_span(name):
                forest, simplified = zip(
                    *map(op_fn, tqdm(forest, desc=name, leave=False, position=idx + 1)), strict=False
                )

            if any(simplified):
                simplification_operation.set(op_id)

            barrier.wait()  # Wait for all workers to finish this operation

            # If simplification has occurred in any worker, stop processing further operations.
            if early_exit and simplification_operation.value != -1:
                break

    return forest, request_id


def create_group(subtree: Tree, group_index: int) -> None:
    """
    Create a group node from a subtree and inserts it into its parent node.

    :param subtree: The subtree to convert into a group.
    :param group_index: The index to use for naming the group.
    """
    label = NodeLabel(NodeType.GROUP, str(group_index))
    subtree.set_label(label)

    new_children = [deepcopy(entity) for entity in subtree.entities()]
    subtree.clear()
    subtree.extend(new_children)


def find_groups(
    equiv_subtrees: TREE_CLUSTER,
    min_support: int,
) -> bool:
    """
    Find and create groups based on the given set of equivalent subtrees.

    :param equiv_subtrees: The set of equivalent subtrees.
    :param min_support: Minimum support of groups.

    :return: A boolean indicating if groups were created.
    """
    frequent_clusters = sorted(
        filter(lambda cluster: len(cluster) > min_support, equiv_subtrees),
        key=lambda cluster: (
            len(cluster),
            sum(len(st.entities()) for st in cluster) / len(cluster),
            sum(st.depth() for st in cluster) / len(cluster),
        ),
        reverse=True,
    )

    group_created = False
    for group_index, subtree_cluster in enumerate(frequent_clusters):
        # Create a group for each subtree in the cluster
        for subtree in subtree_cluster:
            if (
                len(subtree) < 2
                or has_type(subtree)
                or (subtree.parent() and has_type(subtree.parent(), NodeType.GROUP))
                or not all(has_type(node, NodeType.ENT) for node in subtree)
                or subtree.has_duplicate_entity()
            ):
                continue

            create_group(subtree, group_index)
            group_created = True

            group_labels = tuple(sorted({label for subtree in subtree_cluster for label in subtree.entity_labels()}))
            if span := mlflow.get_current_active_span():
                span.add_event(
                    SpanEvent(
                        'create_group',
                        attributes={
                            'group': group_index,
                            'labels': group_labels,
                        },
                    )
                )

    return group_created

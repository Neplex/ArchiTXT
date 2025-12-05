from __future__ import annotations

import ctypes
import functools
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from multiprocessing import Manager, cpu_count
from queue import Full
from typing import TYPE_CHECKING, overload

import mlflow
import more_itertools
from mlflow.entities import SpanEvent
from tqdm.auto import tqdm, trange

from architxt.bucket import TreeBucket
from architxt.metrics import Metrics
from architxt.similarity import DECAY, DEFAULT_METRIC, METRIC_FUNC, TREE_CLUSTER, equiv_cluster
from architxt.tree import Forest, NodeLabel, NodeType, Tree, TreeOID, has_type
from architxt.utils import ExceptionGroup, get_commit_batch_size

from .operations import (
    FindCollectionsOperation,
    FindRelationsOperation,
    FindSubGroupsOperation,
    MergeGroupsOperation,
    Operation,
    ReduceBottomOperation,
    ReduceTopOperation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future
    from multiprocessing.managers import ValueProxy
    from queue import Queue
    from threading import Barrier

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
    decay: float = DECAY,
    epoch: int = 100,
    min_support: int | None = None,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    edit_ops: Sequence[type[Operation]] = DEFAULT_OPERATIONS,
    debug: bool = False,
    max_workers: int | None = None,
    commit: bool | int = True,
    simplify_names: bool = True,
) -> Metrics:
    """
    Rewrite a forest by applying edit operations iteratively.

    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :param epoch: Maximum number of rewriting steps.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param edit_ops: The list of operations to perform on the forest.
    :param debug: Whether to enable debug logging.
    :param max_workers: Number of parallel worker processes to use.
    :param commit: Commit automatically. If already in a transaction, no commit is applied.
        - If False, no commits are made, it relies on the current transaction.
        - If True (default), commits in batch.
        - If an integer, commits every N tree.
        To avoid memory issues, we recommend using incremental commit with large iterables.
        When using TreeBucket, workers always commit in internal transactions (to avoid serialisation).
        The commit parameter only controls the batch size for these commits.
    :param simplify_names: Should the groups/relations names be simplified after the rewrite?

    :return: A `Metrics` object encapsulating the results and metrics calculated for the rewrite process.
    """
    metrics = Metrics(forest, tau=tau, decay=decay, metric=metric)

    if not len(forest):
        return metrics

    batch_size = get_commit_batch_size(commit)
    min_support = min_support or max((len(forest) // 10), 2)
    max_workers = min(len(forest) // batch_size, max_workers or (cpu_count() - 2)) or 1

    if mlflow.active_run():
        mlflow.log_params(
            {
                'nb_sentences': len(forest),
                'tau': tau,
                'decay': decay,
                'epoch': epoch,
                'min_support': min_support,
                'metric': metric.__name__,
                'edit_ops': ', '.join(f"{op_id}: {edit_op.__name__}" for op_id, edit_op in enumerate(edit_ops)),
            }
        )
        metrics.log_to_mlflow(0, debug=debug)

    with (
        mlflow.start_span('rewriting') if mlflow.active_run() else nullcontext(),
        ProcessPoolExecutor(max_workers=max_workers) as executor,
    ):
        for iteration in trange(1, epoch, desc='rewrite trees'):
            with (
                mlflow.start_span(
                    'iteration',
                    attributes={
                        'step': iteration,
                    },
                )
                if mlflow.active_run()
                else nullcontext()
            ):
                has_simplified = _rewrite_step(
                    iteration,
                    forest,
                    tau=tau,
                    decay=decay,
                    min_support=min_support,
                    metric=metric,
                    edit_ops=edit_ops,
                    debug=debug,
                    executor=executor,
                    batch_size=batch_size,
                )

                if mlflow.active_run():
                    metrics.update()
                    metrics.log_to_mlflow(iteration, debug=debug)

                # Stop if no further simplifications are made
                if iteration > 0 and not has_simplified:
                    break

        _post_process(forest, tau=tau, decay=decay, metric=metric, executor=executor, batch_size=batch_size)

        if simplify_names:
            with forest.transaction() if isinstance(forest, TreeBucket) else nullcontext():
                _simplify_names(forest)

        metrics.update()

    if mlflow.active_run():
        metrics.log_to_mlflow(iteration + 1, debug=debug)

    return metrics


def _rewrite_step(
    iteration: int,
    forest: Forest,
    *,
    tau: float,
    decay: float,
    min_support: int,
    metric: METRIC_FUNC,
    edit_ops: Sequence[type[Operation]],
    debug: bool,
    executor: ProcessPoolExecutor,
    batch_size: int,
) -> bool:
    """
    Perform a single rewrite step on the forest.

    :param iteration: The current iteration number.
    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param decay: The similarity decay factor.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param edit_ops: The list of operations to perform on the forest.
    :param debug: Whether to enable debug logging.
    :param executor: A pool executor to parallelize the processing of the forest.
    :param batch_size: The number of trees to process in each batch.

    :return: A flag indicating if any simplifications occurred.
    """
    with (
        mlflow.start_span('reduce_all') if mlflow.active_run() else nullcontext(),
        forest.transaction() if isinstance(forest, TreeBucket) else nullcontext(),
    ):
        for tree in forest:
            tree.reduce_all({NodeType.ENT})

    with mlflow.start_span('equiv_cluster') if mlflow.active_run() else nullcontext():
        equiv_subtrees = equiv_cluster(forest, tau=tau, decay=decay, metric=metric, _step=iteration if debug else None)

    with (
        mlflow.start_span('find_groups') if mlflow.active_run() else nullcontext(),
        forest.transaction() if isinstance(forest, TreeBucket) else nullcontext(),
    ):
        find_groups(equiv_subtrees, min_support)

    op_id = apply_operations(
        [operation(tau=tau, decay=decay, min_support=min_support, metric=metric) for operation in edit_ops],
        forest,
        batch_size=batch_size,
        equiv_subtrees=equiv_subtrees,
        executor=executor,
    )

    if mlflow.active_run() and op_id is not None:
        mlflow.log_metric('edit_op', op_id, step=iteration)

    return op_id is not None


def _post_process(
    forest: Forest,
    *,
    tau: float,
    decay: float,
    metric: METRIC_FUNC,
    executor: ProcessPoolExecutor,
    batch_size: int,
) -> None:
    """
    Post-process the forest to find and name relations and collections.

    :param forest: The forest to perform on.
    :param tau: Threshold for subtree similarity when clustering.
    :param decay: The similarity decay factor.
    :param metric: The metric function used to compute similarity between subtrees.
    :param executor: A pool executor to parallelize the processing of the forest.
    :param batch_size: The number of trees to process in each batch.
    """
    equiv_subtrees = equiv_cluster(forest, tau=tau, decay=decay, metric=metric)

    apply_operations(
        [
            (
                '[post-process] name_relations',
                FindRelationsOperation(tau=tau, decay=decay, min_support=0, metric=metric, naming_only=True),
            ),
            (
                '[post-process] name_collections',
                FindCollectionsOperation(tau=tau, decay=decay, min_support=0, metric=metric, naming_only=True),
            ),
        ],
        forest,
        equiv_subtrees=equiv_subtrees,
        early_exit=False,
        executor=executor,
        batch_size=batch_size,
    )


def _simplify_names(forest: Forest) -> None:
    """
    Simplify names in the forest by reducing complex labels to more readable forms.

    :param forest: The forest to simplify names in
    """
    simplified_to_labels: defaultdict[str, list[str]] = defaultdict(list)  # simplified_name -> [original_labels]

    for tree in forest:
        for subtree in tree.subtrees():
            if not has_type(subtree, NodeType.GROUP):
                continue

            simplified_name = _get_base_name(subtree.label.name)
            label_list = simplified_to_labels[simplified_name]

            if subtree.label.name not in label_list:
                label_list.append(subtree.label.name)

            if (idx := label_list.index(subtree.label.name)) > 0:
                simplified_name = f'{simplified_name}_{idx}'

            subtree.label = NodeLabel(NodeType.GROUP, simplified_name)

    for tree in forest:
        for subtree in tree.subtrees():
            if has_type(subtree, NodeType.REL) and (groups := subtree.groups()):
                subtree.label = NodeLabel(NodeType.REL, '<->'.join(sorted(groups)))

            if has_type(subtree, NodeType.COLL) and (groups := subtree.groups()):
                subtree.label = NodeLabel(NodeType.COLL, groups.pop())


def _get_base_name(name: str) -> str:
    """
    Get the simplified name for a group subtree.

    Returns either:
    - The base name (extracted from label like "Territories_1_5_2" -> "Territories")
    - Generic "UndefinedGroup" for group without a base name
    """
    base_name = re.sub(r'(_\d+)+$', '', name)

    if re.match(r'^\d*$', base_name):
        base_name = 'UndefinedGroup'

    return base_name


def apply_operations(
    edit_ops: Sequence[Operation | tuple[str, Operation]],
    forest: Forest,
    *,
    equiv_subtrees: TREE_CLUSTER,
    early_exit: bool = True,
    executor: ProcessPoolExecutor,
    batch_size: int,
) -> int | None:
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
    :param batch_size: The number of trees to process in each batch.

    :return: The index of the operation that successfully simplified a tree, or `None` if no operation succeeded.
    """
    if not edit_ops:
        return None

    run_id = run.info.run_id if (run := mlflow.active_run()) else None
    edit_ops_names = [(op.name, op) if isinstance(op, Operation) else op for op in edit_ops]
    workers_count = executor._max_workers
    futures: list[Future]

    with Manager() as manager:
        shared_equiv = manager.Value(ctypes.py_object, equiv_subtrees, lock=False)
        simplification_operation = manager.Value(ctypes.c_int, -1, lock=False)
        barrier = manager.Barrier(workers_count + isinstance(forest, TreeBucket))
        queue: Queue[TreeOID | None] | None = (
            manager.Queue(maxsize=workers_count * 3) if isinstance(forest, TreeBucket) else None
        )

        worker_fn = functools.partial(
            _apply_operations_worker,
            edit_ops=edit_ops_names,
            shared_equiv_subtrees=shared_equiv,
            early_exit=early_exit,
            simplification_operation=simplification_operation,
            barrier=barrier,
            run_id=run_id,
            batch_size=batch_size,
            queue=queue,
        )

        if queue is not None and isinstance(forest, TreeBucket):
            futures = [executor.submit(worker_fn, idx=idx, forest=forest) for idx in range(workers_count)]
            _fill_queue(futures, forest, simplification_operation, barrier, queue, edit_ops_names, early_exit)

        else:
            futures = [
                executor.submit(worker_fn, idx=idx, forest=tuple(batch))
                for idx, batch in enumerate(more_itertools.distribute(workers_count, forest))
            ]

        new_forest = []
        for future in as_completed(futures):
            request_id, trees = future.result()

            if trees:
                new_forest.extend(trees)

            if worker_trace := mlflow.get_trace(request_id):
                mlflow.add_trace(worker_trace)

        if new_forest:
            forest[:] = new_forest

        op_id = simplification_operation.get()

    return op_id if op_id >= 0 else None


def _check_worker_health(futures: Sequence[Future], barrier: Barrier, error: Exception | None = None) -> None:
    if errors := [exc for future in futures if not future.running() and (exc := future.exception())]:
        barrier.abort()

        msg = 'Some workers has failed'
        raise ExceptionGroup(msg, errors) from error


def _fill_queue(
    futures: Sequence[Future],
    forest: TreeBucket,
    simplification_operation: ValueProxy[int],
    barrier: Barrier,
    queue: Queue[TreeOID | None],
    edit_ops: Sequence[tuple[str, Operation]],
    early_exit: bool,
    timeout: int = 1,
) -> None:
    for op_name, _ in edit_ops:
        # Refill the queue with a new batch of object IDs from the bucket.
        # This avoids holding all IDs in memory at once, which could cause OOM issues.
        for oid in tqdm(forest.oids(), total=len(forest), desc=op_name, leave=False):
            while True:
                try:
                    queue.put(oid, timeout=timeout)
                    break

                except Full as error:
                    # As we wait, we check for worker failures to avoid deadlocks.
                    _check_worker_health(futures, barrier, error)

        # Signal to each worker that the current batch is complete.
        # One sentinel per worker to ensure a clean exit or sync point.
        for _ in range(len(futures)):
            queue.put(None)

        # Synchronize with all worker processes:
        # - If a simplification occurred, workers will exit early.
        # - Otherwise, the main process will continue to the next batch.
        while True:
            try:
                barrier.wait(timeout=timeout)
                break

            except TimeoutError as error:
                # As we wait, we check for worker failures to avoid deadlocks.
                _check_worker_health(futures, barrier, error)

        # If simplification has occurred in any worker, stop processing further operations.
        if early_exit and simplification_operation.value != -1:
            break

    # Synchronize the bucket to ensure all changes are visible to the main process and avoid caching issue.
    # NOTE: It may not be required, but it is fool-proof
    forest.sync()


@overload
def _apply_operations_worker(
    idx: int,
    edit_ops: Sequence[tuple[str, Operation]],
    forest: TreeBucket,
    queue: Queue[TreeOID | None],
    shared_equiv_subtrees: ValueProxy[TREE_CLUSTER],
    early_exit: bool,
    simplification_operation: ValueProxy[int],
    barrier: Barrier,
    run_id: str | None,
    batch_size: int,
) -> tuple[str, None]: ...


@overload
def _apply_operations_worker(
    idx: int,
    edit_ops: Sequence[tuple[str, Operation]],
    forest: Forest,
    queue: None,
    shared_equiv_subtrees: ValueProxy[TREE_CLUSTER],
    early_exit: bool,
    simplification_operation: ValueProxy[int],
    barrier: Barrier,
    run_id: str | None,
    batch_size: int,
) -> tuple[str, Forest]: ...


def _apply_operations_worker(
    idx: int,
    edit_ops: Sequence[tuple[str, Operation]],
    forest: TreeBucket | Forest,
    queue: Queue[TreeOID | None] | None,
    shared_equiv_subtrees: ValueProxy[TREE_CLUSTER],
    early_exit: bool,
    simplification_operation: ValueProxy[int],
    barrier: Barrier,
    run_id: str | None,
    batch_size: int,
) -> tuple[str, Forest | None]:
    """
    Apply the given operations to a forest.

    - In TreeBucket mode, tree IDs are consumed from a shared queue.
      A sentinel value (None) signals the end of the input.
      Each worker processes trees incrementally and commits changes in batches.
    - In list-based mode (plain Forest), the entire collection is processed in a single pass.

    Workers synchronize using a barrier after each operation.

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
    :param forest: The forest or bucket of trees to process.
    :param queue: A shared queue of tree IDs still waiting to be processed (for bucket use).
    :param shared_equiv_subtrees: The shared set of equivalent subtrees.
    :param early_exit: A boolean flag indicating whether to stop after the first successful operation.
                       If `False`, all operations are applied.
    :param simplification_operation: A shared integer value to store the index of the operation that simplified a tree.
    :param barrier: A barrier to synchronize the workers before starting the next operation.
    :param run_id: The Mlflow run_id to link to.
    :param batch_size: Number of trees to process before committing changes in bucket mode.
    :return: Tuple of MLflow request ID and modified forest, or None if using a bucket.
    """
    equiv_subtrees = shared_equiv_subtrees.get()

    with (
        forest if isinstance(forest, TreeBucket) else nullcontext(),
        mlflow.start_run(run_id=run_id) if run_id else nullcontext(),
        mlflow.start_span('worker', attributes={'worker_id': idx}) as span,
    ):
        request_id = span.request_id

        for op_id, (name, operation) in enumerate(edit_ops):
            op_fn = functools.partial(operation.apply, equiv_subtrees=equiv_subtrees)

            with mlflow.start_span(name):
                if queue is None:
                    # List-based mode: apply operation directly
                    forest_iterator = tqdm(forest, desc=name, total=len(forest), leave=False, position=idx + 1)
                    simplified = any(map(op_fn, forest_iterator))

                elif isinstance(forest, TreeBucket):
                    # Bucket-based mode: apply operation for each tree in the queue and commit in batches
                    simplified = False
                    modifications_in_transaction = 0

                    with ExitStack() as transaction_stack:
                        transaction_stack.enter_context(forest.transaction())

                        while (oid := queue.get()) is not None:
                            modified = op_fn(forest[oid])
                            modifications_in_transaction += modified
                            simplified |= modified

                            # Commit in batches to bound memory usage and enable cache clearing
                            if modifications_in_transaction >= batch_size:
                                transaction_stack.close()  # Commit current transaction
                                transaction_stack.enter_context(forest.transaction())  # Begin new transaction
                                modifications_in_transaction = 0

                else:
                    msg = 'Using an in-memory collection may result in excessive serialization when used with a queue.'
                    raise ValueError(msg)

            if simplified:
                simplification_operation.set(op_id)

            barrier.wait()  # Wait for all workers to finish this operation

            if isinstance(forest, TreeBucket):
                # Sync forest to avoid any cache issue between operations
                # NOTE: It may not be required, but it is fool-proof
                forest.sync()

            # If simplification has occurred in any worker, stop processing further operations.
            if early_exit and simplification_operation.value != -1:
                break

    return request_id, None if isinstance(forest, TreeBucket) else forest


def create_group(subtree: Tree, group_name: str) -> None:
    """
    Create a group node from a subtree and inserts it into its parent node.

    :param subtree: The subtree to convert into a group.
    :param group_name: The name to use for the group.
    """
    subtree.label = NodeLabel(NodeType.GROUP, group_name)
    subtree[:] = [entity.detach() for entity in subtree.entities()]


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
        filter(lambda cluster_name: len(equiv_subtrees[cluster_name]) > min_support, equiv_subtrees.keys()),
        key=lambda cluster_name: (
            len(equiv_subtrees[cluster_name]),
            sum(len(st.entities()) for st in equiv_subtrees[cluster_name]) / len(equiv_subtrees[cluster_name]),
            sum(st.depth for st in equiv_subtrees[cluster_name]) / len(equiv_subtrees[cluster_name]),
        ),
        reverse=True,
    )

    group_created = False
    for cluster_name in frequent_clusters:
        subtree_cluster = equiv_subtrees[cluster_name]

        # Create a group for each subtree in the cluster
        for subtree in subtree_cluster:
            if (
                len(subtree) < 2
                or (subtree.parent and has_type(subtree.parent, NodeType.GROUP))
                or not all(has_type(node, NodeType.ENT) for node in subtree)
                or subtree.has_duplicate_entity()
            ):
                continue

            if has_type(subtree, NodeType.GROUP):  # Renaming only
                subtree.label = NodeLabel(NodeType.GROUP, cluster_name)
                continue

            create_group(subtree, cluster_name)
            group_created = True

            if span := mlflow.get_current_active_span():
                group_labels = tuple(
                    sorted({label for subtree in subtree_cluster for label in subtree.entity_labels()})
                )
                span.add_event(
                    SpanEvent(
                        'create_group',
                        attributes={
                            'group': cluster_name,
                            'labels': group_labels,
                        },
                    )
                )

    return group_created

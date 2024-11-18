import functools
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy

import mlflow
from nltk import Production, TreePrettyPrinter
from pqdm.processes import pqdm
from tqdm import trange

from architxt import operations
from architxt.model import NodeType
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC, TREE_CLUSTER, equiv_cluster
from architxt.tree import Forest, Tree, has_type, reduce_all

DEFAULT_OPERATIONS = (
    operations.find_subgroups,
    operations.merge_groups,
    operations.find_collections,
    operations.find_relationship,
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
    n_jobs: int = 6,
) -> Forest:
    min_support = min_support or max((len(forest) // 10), 2)
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
                    n_jobs=n_jobs,
                )

                if iteration != 0 and not has_simplified:
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
    n_jobs: int,
):
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
        operation = functools.partial(
            edit_op, equiv_subtrees=equiv_subtrees, tau=tau, min_support=min_support, metric=metric
        )

        with mlflow.start_span(edit_op.__name__):
            result = pqdm(
                forest,
                operation,
                n_jobs=n_jobs,
                exception_behaviour='immediate',
                leave=False,
                desc=edit_op.__name__,
            )

        forest, simplified = zip(*result)
        if has_simplified := any(simplified):
            mlflow.log_metric('edit_op', op_id, step=iteration)
            break

    _post_process(forest, tau, metric)

    # _log_productions(rooted_forest)
    _log_metrics(iteration, forest, equiv_subtrees)

    return forest, has_simplified


def _post_process(forest: Forest, tau: float, metric: METRIC_FUNC) -> None:
    equiv_subtrees = equiv_cluster(forest, tau=tau, metric=metric)

    forest, _ = zip(
        *(
            operations.find_relationship(
                tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees, naming_only=True
            )
            for tree in forest
        )
    )

    forest, _ = zip(
        *(
            operations.find_collections(
                tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees, naming_only=True
            )
            for tree in forest
        )
    )


def _get_productions(forest: Forest) -> Counter[str]:
    """
    Extracts and counts non-lexical productions from a forest.

    :param forest: A forest to extract productions from.
    :return: A `Counter` object containing the counts of each production rule.
    """
    production_counter: Counter[str] = Counter()

    for tree in forest:
        for production in tree.productions():
            if production.is_lexical():
                continue  # Skip lexical productions

            lhs = production.lhs().symbol()
            rhs = sorted(symbol.symbol() for symbol in production.rhs())
            rhs_str = f'{rhs[0]}+' if has_type(lhs, NodeType.COLL) else ' '.join(rhs)

            production_rule = f'{lhs} -> {rhs_str}'
            production_counter.update([production_rule])

    return production_counter


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
    # Compute the number of production rules in the forest
    num_productions = len(_get_productions(forest))

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
            'num_productions': num_productions,
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


def _log_productions(root_tree: Tree):
    production_count = Counter(root_tree.productions())
    for production, count in production_count.most_common():
        production: Production
        if production.is_nonlexical():
            # stream.write(f'[{count}] {production}\n')
            pass

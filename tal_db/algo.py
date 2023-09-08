import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from typing import IO

import mlflow
from nltk import Production
from tqdm import trange

from . import operations, tree
from .model import NodeType
from .similarity import similarity, equiv_cluster, METRIC_FUNC, DEFAULT_METRIC, TREE_CLUSTER
from .tree import has_type

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
        root_tree: tree.ParentedTree, *,
        tau: float = 0.7,
        epoch: int = 100,
        min_support: int | None = None,
        metric: METRIC_FUNC = DEFAULT_METRIC,
        edit_ops: Sequence[operations.OPERATION] = DEFAULT_OPERATIONS,
        stream: IO[str] = sys.stdout
) -> None:
    min_support = min_support or max((len(root_tree) // 10), 2)
    mlflow.log_params({
        'tau': tau,
        'epoch': epoch,
        'min_support': min_support,
        'metric': metric.__name__,
        'edit_ops': ','.join(op.__name__ for op in edit_ops)
    })

    stream.write(
        f'Params:\n'
        f'tau: {tau}\n'
        f'epoch: {epoch}\n'
        f'min_support: {min_support}\n'
        f'metric: {metric.__name__}\n'
        f'edit_ops: {[op.__name__ for op in edit_ops]}\n\n'
    )
    stream.write('== Init ====================\n')
    root_tree.pretty_print(stream=stream)

    _log_metrics(0, root_tree)

    iteration = 0
    for iteration in trange(epoch, desc='rewrite trees'):
        stream.write(f'== Iteration: {iteration} ====================\n')
        root_tree.pretty_print(stream=stream)
        similarity.cache_clear()

        _pre_process(root_tree)

        equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)
        _display_clusters(equiv_subtrees, stream)

        operations.find_groups(root_tree, equiv_subtrees, tau=tau, min_support=min_support, metric=metric)

        has_simplified = False
        for op_id, edit_op in enumerate(edit_ops):
            stream.write(f'Run: {edit_op.__name__}\n')
            if has_simplified := edit_op(root_tree, equiv_subtrees, tau, min_support, metric):
                mlflow.log_metric('edit_op', op_id, step=iteration + 1)
                break

        _display_productions(root_tree, stream)
        _log_metrics(iteration + 1, root_tree, equiv_subtrees)

        if not has_simplified:
            break

    # Post-processing to uniformize the rules
    equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)
    while operations.find_subgroups(root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees):
        equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)

    while operations.find_relationship(root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees):
        continue

    while operations.find_collections(root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees):
        continue

    _post_process(root_tree, equiv_subtrees)
    _display_productions(root_tree, stream)
    _log_metrics(iteration + 2, root_tree, equiv_subtrees)


def _pre_process(root_tree: tree.ParentedTree) -> None:
    for subtree in root_tree.subtrees(lambda x: x != root_tree.root()):
        if not has_type(subtree, NodeType.ENT):
            subtree.set_label(uuid.uuid4().hex[:8])
        if subtree.height() > 2 and not tree.has_type(subtree, NodeType.ENT):
            tree.reduce(subtree.parent(), subtree.parent_index())


def _post_process(root_tree: tree.ParentedTree, equiv_subtrees: TREE_CLUSTER) -> None:
    # for cluster in equiv_subtrees:
    #     for sub_tree in cluster:
    #         if has_type(sub_tree, NodeType.GROUP):
    #             pass
    pass


def _log_metrics(iteration: int, root_tree: tree.Tree, equiv_subtrees: TREE_CLUSTER = ()):
    production_count = Counter(root_tree.productions())
    prod_keys = production_count.keys()

    nb_ent = sum(has_type(p, NodeType.ENT) for p in prod_keys)
    nb_group = sum(has_type(p, NodeType.GROUP) for p in prod_keys)
    nb_coll = sum(has_type(p, NodeType.COLL) for p in prod_keys)
    nb_rel = sum(has_type(p, NodeType.REL) for p in prod_keys)
    nb_data = sum(p[1] for p in production_count.items() if has_type(p[0], NodeType.ENT))
    nb_group_instance = sum(p[1] for p in production_count.items() if has_type(p[0], NodeType.GROUP))
    group_ratio = nb_group_instance / nb_group if nb_group != 0 else 0

    mlflow.log_metrics({
        'nb_prod': len(production_count),
        'nb_ent': nb_ent,
        'nb_data': nb_data,
        'nb_group': nb_group,
        'nb_coll': nb_coll,
        'nb_rel': nb_rel,
        'group_ratio': group_ratio,
        'nb_equiv_subtrees': len(equiv_subtrees),
    }, step=iteration)


def _display_clusters(equiv_subtrees: TREE_CLUSTER, stream: IO[str]):
    clusters_info = [f'clusters (count, max_len, elems): {len(equiv_subtrees)}\n']
    for equiv_class in sorted(equiv_subtrees, key=lambda x: -len(x)):
        cluster = [equiv_tree.treeposition() for equiv_tree in equiv_class]
        max_len = max(len(equiv_tree.leaves()) for equiv_tree in equiv_class)

        cluster_info = f'- {len(cluster)} {max_len} {cluster}\n'
        clusters_info.append(cluster_info)

    stream.writelines(clusters_info)


def _display_productions(root_tree: tree.Tree, stream: IO[str]):
    stream.write('Result:\n')
    root_tree.pretty_print(stream=stream)
    production_count = Counter(root_tree.productions())
    for production, count in production_count.most_common():
        production: Production
        if production.is_nonlexical():
            stream.write(f'[{count}] {production}\n')

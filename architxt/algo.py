import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from typing import IO

import mlflow
from nltk import Production
from tqdm import tqdm, trange

from architxt import operations, tree
from architxt.model import NodeType
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC, TREE_CLUSTER, equiv_cluster, get_equiv_of, similarity
from architxt.tree import has_type

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
    root_tree: tree.ParentedTree,
    *,
    tau: float = 0.7,
    epoch: int = 100,
    min_support: int | None = None,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    edit_ops: Sequence[operations.OPERATION] = DEFAULT_OPERATIONS,
    stream: IO[str] = sys.stdout,
) -> tree.ParentedTree:
    min_support = min_support or max((len(root_tree) // 10), 2)
    mlflow.log_params(
        {
            'tau': tau,
            'epoch': epoch,
            'min_support': min_support,
            'metric': metric.__name__,
            'edit_ops': ', '.join(f"{op_id}: {edit_op.__name__}" for op_id, edit_op in enumerate(edit_ops)),
        }
    )

    stream.write(
        f'Params:\n'
        f'tau: {tau}\n'
        f'epoch: {epoch}\n'
        f'min_support: {min_support}\n'
        f'metric: {metric.__name__}\n'
        f'edit_ops: {[f"{op_id}: {edit_op.__name__}" for op_id, edit_op in enumerate(edit_ops)]}\n\n'
    )
    stream.write('== Init ====================\n')
    root_tree.pretty_print(stream=stream)

    for iteration in trange(epoch, desc='rewrite trees'):
        stream.write(f'== Iteration: {iteration} ====================\n')
        root_tree.pretty_print(stream=stream)
        stream.flush()
        similarity.cache_clear()
        get_equiv_of.cache_clear()

        _pre_process(root_tree)

        equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)
        _display_clusters(equiv_subtrees, stream)

        operations.find_groups(root_tree, equiv_subtrees, tau=tau, min_support=min_support, metric=metric)

        if iteration > 0:
            has_simplified = False
            for op_id, edit_op in enumerate(edit_ops):
                stream.write(f'Run: {edit_op.__name__}\n')
                if has_simplified := edit_op(root_tree, equiv_subtrees, tau, min_support, metric):
                    mlflow.log_metric('edit_op', op_id, step=iteration + 1)
                    break

        for _ in range(3):
            operations.find_relationship(root_tree, equiv_subtrees, tau, min_support, metric, naming_only=True)
            operations.find_collections(root_tree, equiv_subtrees, tau, min_support, metric, naming_only=True)

        _display_productions(root_tree, stream)
        _log_metrics(iteration, root_tree, equiv_subtrees)

        if iteration != 0 and not has_simplified:
            break

    # Post-processing to uniformize the rules
    # _post_process(root_tree, equiv_subtrees, tau=tau, metric=metric)

    grammar = ""
    for production, count in _get_productions(root_tree).items():
        grammar += f'[{count}] {production}\n'

    mlflow.log_text(grammar, 'grammar.txt')

    _display_productions(root_tree, stream)
    # _log_metrics(iteration + 2, root_tree, equiv_subtrees)
    return root_tree


def _pre_process(root_tree: tree.ParentedTree) -> None:
    for subtree in tqdm(list(root_tree.subtrees(lambda x: x != root_tree.root())), desc='pre_process', leave=False):
        if not has_type(subtree, NodeType.ENT):
            subtree.set_label(uuid.uuid4().hex[:8])
        if subtree.height() > 2 and not tree.has_type(subtree, NodeType.ENT):
            tree.reduce(subtree.parent(), subtree.parent_index())


def _post_process(root_tree: tree.ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, metric: METRIC_FUNC) -> None:
    tau /= 2
    equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)

    print('post processing')

    while operations.find_subgroups(root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees):
        equiv_subtrees = equiv_cluster(root_tree, tau=tau, metric=metric)

    while operations.find_relationship(
        root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees, naming_only=True
    ):
        continue

    while operations.find_collections(root_tree, tau=tau, min_support=0, metric=metric, equiv_subtrees=equiv_subtrees):
        continue


def _get_productions(root_tree: tree.Tree) -> Counter:
    counter = Counter()

    for production in root_tree.productions():
        if production.is_nonlexical():
            lhs = production.lhs().symbol()
            rhs = sorted(x.symbol() for x in production.rhs())
            rhs_str = f'{rhs[0]}+' if has_type(lhs, NodeType.COLL) else ' '.join(rhs)

            counter.update([f'{lhs} -> {rhs_str}'])

    return counter


def _log_metrics(iteration: int, root_tree: tree.Tree, equiv_subtrees: TREE_CLUSTER = ()):
    nb_prod = len(_get_productions(root_tree))
    labels = {subtree.label() for subtree in root_tree.subtrees()}
    nb_unlabelled = sum(not has_type(x) for x in root_tree.subtrees())

    nb_ent = sum(has_type(x, NodeType.ENT) for x in labels)
    nb_ent_instance = sum(has_type(x, NodeType.ENT) for x in root_tree.subtrees())
    ent_ratio = nb_ent_instance / nb_ent if nb_ent != 0 else 0

    nb_group = sum(has_type(x, NodeType.GROUP) for x in labels)
    nb_group_instance = sum(has_type(x, NodeType.GROUP) for x in root_tree.subtrees())
    group_ratio = nb_group_instance / nb_group if nb_group != 0 else 0

    nb_coll = sum(has_type(x, NodeType.COLL) for x in labels)
    nb_coll_instance = sum(has_type(x, NodeType.COLL) for x in root_tree.subtrees())
    coll_ratio = nb_coll_instance / nb_coll if nb_coll != 0 else 0

    nb_rel = sum(has_type(x, NodeType.REL) for x in labels)
    nb_rel_instance = sum(has_type(x, NodeType.REL) for x in root_tree.subtrees())
    rel_ratio = nb_rel_instance / nb_rel if nb_rel != 0 else 0

    mlflow.log_metrics(
        {
            'nb_prod': nb_prod,
            'nb_non_terminal': len(labels),
            'nb_unlabelled_node': nb_unlabelled,
            'nb_equiv_subtrees': len(equiv_subtrees),
            'nb_ent': nb_ent,
            'nb_ent_instance': nb_ent_instance,
            'ent_ratio': ent_ratio,
            'nb_group': nb_group,
            'nb_group_instance': nb_group_instance,
            'group_ratio': group_ratio,
            'nb_coll': nb_coll,
            'nb_coll_instance': nb_coll_instance,
            'coll_ratio': coll_ratio,
            'nb_rel': nb_rel,
            'nb_rel_instance': nb_rel_instance,
            'rel_ratio': rel_ratio,
        },
        step=iteration,
    )


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

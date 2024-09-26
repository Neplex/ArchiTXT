import math
from collections import Counter
from collections import deque
from collections.abc import Callable
from itertools import combinations, groupby

from joblib import Parallel, delayed
from tqdm import tqdm

from architxt.model import NodeType, NodeLabel
from architxt.similarity import METRIC_FUNC, TREE_CLUSTER, get_equiv_of
from architxt.tree import ParentedTree, has_type, Tree, ins_elem, del_elem, update_cache

__all__ = [
    'OPERATION',
    'find_groups', 'find_subgroups', 'merge_groups',
    'find_relationship', 'find_collections',
    'reduce_bottom', 'reduce_top', 'merge_sentences',
]

OPERATION = Callable[[ParentedTree, TREE_CLUSTER, float, int, METRIC_FUNC], bool]
TASK_POOL = Parallel(n_jobs=-2, require='sharedmem', return_as='generator', batch_size=1)

trace = open('trace.txt', 'w')


def reduce_bottom(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    # return any([
    #     reduce(subtree.parent(), subtree.parent_index(), set(NodeType))
    #     for subtree in t.subtrees(lambda x: not has_type(x) and any(has_type(child, NodeType.ENT) for child in x))
    # ])

    for subtree in reversed(list(t.subtrees(
            lambda x: not has_type(x) and any(has_type(child, NodeType.ENT) for child in x) and x.height() < (t.height() - 1)))):
        parent = subtree.parent()

        if not parent:
            continue

        position = subtree.treeposition()
        label = subtree.label()
        old = [child.label() for child in parent]

        new_children = [Tree.convert(child) for child in subtree]

        parent_pos = subtree.parent_index()
        parent.pop(parent_pos)

        for child in reversed(new_children):
            ins_elem(parent, child, parent_pos)

        new = [child.label() for child in parent]
        print(
            f'reduce_bottom {label} ({position}) {old} -> {new}',
            file=trace
        )
        return True

    return False


def reduce_top(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC, level: int = 2) -> bool:
    # return any([
    #     reduce(subtree.parent(), subtree.parent_index(), set(NodeType))
    #     for subtree in t.subtrees(lambda x: not has_type(x) and x.height() == (t.height() - 1))
    # ])
    reduced = False

    for subtree in t.subtrees(lambda x: not has_type(x) and x.height() == (t.height() - level)):
        parent = subtree.parent()
        position = subtree.treeposition()
        label = subtree.label()
        old = [child.label() for child in parent]

        new_children = [Tree.convert(child) for child in subtree]

        parent_pos = subtree.parent_index()
        parent.pop(parent_pos)

        for child in reversed(new_children):
            ins_elem(parent, child, parent_pos)

        new = [child.label() for child in parent]
        print(
            f'reduce_top {label} ({position}) {old} -> {new}',
            file=trace
        )
        reduced = True

    return reduced


def merge_sentences(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    return reduce_top(t, equiv_subtrees, tau, min_support, metric, 1)


def _create_group(subtree: ParentedTree, k: int) -> None:
    entity_trees = (Tree.convert(ent_tree) for ent_tree in subtree.subtrees(lambda x: has_type(x, NodeType.ENT)))
    group_tree = Tree(NodeLabel(NodeType.GROUP, str(k)), children=entity_trees)

    group_parent = subtree.parent()
    group_pos = subtree.parent_index()
    group_parent.pop(group_pos)
    ins_elem(group_parent, group_tree, group_pos)


def find_groups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC):
    frequent_subtrees = filter(lambda x: len(x) >= min_support, equiv_subtrees)
    frequent_subtrees = sorted(frequent_subtrees, key=lambda x: sum((y.depth() / (len(y) or 1)) for y in x) / len(x))

    for k, min_group in enumerate(tqdm(frequent_subtrees, desc='find groups', leave=False)):
        # Skip frequent subtrees containing or contain in groups
        if any(has_type(x, NodeType.GROUP) or has_type(subtree, NodeType.GROUP) for subtree in min_group for x in subtree):
            continue

        deque(TASK_POOL(
            delayed(_create_group)(subtree, k)
            for subtree in tqdm(min_group, leave=False)
        ), maxlen=0)

    print('=' * 50, file=trace)
    trace.flush()


def _find_subgroups_inner(
        subtree: ParentedTree, sub_group: tuple[ParentedTree, ...],
        equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[ParentedTree, int] | None:
    if max(Counter(x.label() for x in sub_group).values()) > 1:
        return None

    # Copy the tree
    new_tree = subtree.root().copy(deep=True)
    new_subtree = new_tree[subtree.treeposition()]

    group_tree = Tree(NodeLabel(NodeType.GROUP), children=[Tree.convert(ent_tree) for ent_tree in sub_group])

    for ent_tree in sorted(sub_group, key=lambda x: x.parent_index(), reverse=True):
        del_elem(new_subtree, ent_tree.parent_index())

    group_pos = min(ent_tree.parent_index() for ent_tree in sub_group)
    ins_elem(new_subtree, group_tree, group_pos)

    # Compute equivalent class
    equiv_group = get_equiv_of(new_subtree[group_pos], equiv_subtrees, tau=tau, metric=metric)
    support = len(equiv_group)

    if support <= min_support:
        return None

    return new_subtree, support


def find_subgroups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False

    for subtree in tqdm(list(reversed(list(t.subtrees(
            lambda x: not isinstance(x, str) and x != t.root() and x.parent() != t.root() and not has_type(x) and any(has_type(y, NodeType.ENT) for y in x)
    )))), desc='find subgroups', leave=False):
        group_support = len(get_equiv_of(subtree, equiv_subtrees, tau=tau, metric=metric))
        entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), subtree))
        parent_idx = subtree.parent_index()
        parent = subtree.parent()

        k = len(entity_trees)

        if len(entity_trees) == len(subtree):
            k -= 1

        with tqdm(desc='k-groups', total=k, leave=False) as pbar:
            while k > 1:
                pbar.update(k)

                # Get k-subgroup with maximum support
                nb_combinations = math.comb(len(entity_trees), k)
                k_groups = combinations(entity_trees, k)
                k_groups_support = TASK_POOL(
                    delayed(_find_subgroups_inner)(
                        subtree, sub_group, equiv_subtrees=equiv_subtrees, tau=tau, min_support=group_support,
                        metric=metric
                    )
                    for sub_group in
                    tqdm(k_groups, leave=False, total=nb_combinations, desc=f'{subtree.label()} n={len(entity_trees)} k={k}')
                )

                # Compute max merge
                max_subtree, _ = max(
                    filter(lambda x: x is not None, k_groups_support),
                    key=lambda x: x[1],
                    default=(None, None)
                )

                # If no k-group found, we reduce group size
                if not max_subtree:
                    k -= 1
                    continue

                # A group is found, we need to add the new subgroup tree
                print(
                    f'find_subgroups {subtree.treeposition()} k={k}:\t{[x.label() for x in subtree]} -> {[x.label() for x in max_subtree]}',
                    file=trace
                )
                trace.flush()
                simplified = True

                # Replace subtree with the newly constructed one
                parent.pop(parent_idx)
                parent.insert(parent_idx, ParentedTree.convert(Tree.convert(max_subtree)))
                update_cache(parent[parent_idx])

                # Remove used entity trees and start over
                entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), max_subtree))

                # Keep searching k-group (reduce k if the number of remaining entities is lower than current k)
                k = min(len(entity_trees), k)

    return simplified


def _merge_groups_inner(
        subtree: ParentedTree, combined_groups: tuple[ParentedTree, ...],
        equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[ParentedTree | None, int]:
    sub_group = []
    max_sub_group_support = 0
    nb_group = 0
    group_name: NodeLabel

    for group_ent in combined_groups:
        if has_type(group_ent, NodeType.ENT):
            sub_group.append(group_ent)

        elif has_type(group_ent, NodeType.GROUP):
            if len(group_ent) == 1:  # Group of size 1 are treated as entities
                sub_group.append(group_ent[0])

            else:
                nb_group += 1
                group_name = group_ent.label()
                sub_group_support = len(get_equiv_of(group_ent, equiv_subtrees, tau=tau, metric=metric))
                if sub_group_support > max_sub_group_support:
                    max_sub_group_support = sub_group_support

                sub_group.extend(group_ent.entities())

    # Skip invalid groups with duplicate entities
    # assert all(has_type(ent, NodeType.ENT) for ent in sub_group)
    if not all(has_type(ent, NodeType.ENT) for ent in sub_group) or not sub_group or nb_group != 1 or max(Counter(x.label() for x in sub_group).values()):
        return None, 0

    # Copy the tree
    new_tree = subtree.root().copy(deep=True)
    new_subtree = new_tree[subtree.treeposition()]

    # Create new tree version
    group_tree = Tree(group_name, children=[Tree.convert(ent_tree) for ent_tree in sub_group])
    group_pos = min(group_ent.parent_index() for group_ent in combined_groups)

    for group_ent in sorted(combined_groups, key=lambda x: x.parent_index(), reverse=True):
        new_subtree.pop(group_ent.parent_index())
    ins_elem(new_subtree, group_tree, group_pos)

    # Compute equivalent class
    equiv_group = get_equiv_of(new_subtree[group_pos], equiv_subtrees, tau=tau, metric=metric)
    support = len(equiv_group)

    if support < min_support:  # or support < max_sub_group_support:
        return None, 0

    if equiv_group and has_type(equiv_group[0], NodeType.GROUP):
        new_subtree[group_pos].set_label(equiv_group[0].label())

    return new_subtree, support


def merge_groups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False

    for subtree in tqdm(list(reversed(list(t.subtrees(
            lambda x: not isinstance(x, str) and x != t.root() and x.parent() != t.root() and not has_type(x) and any(has_type(y, NodeType.GROUP) for y in x)
    )))), desc='merge groups', leave=False):
        # group_support = len(get_equiv_of(subtree, equiv_subtrees, tau=tau, metric=metric))
        group_ent_trees = list(filter(lambda x: has_type(x, {NodeType.GROUP, NodeType.ENT}) or not has_type(x), subtree))
        parent = subtree.parent()
        parent_idx = subtree.parent_index()

        k = len({get_equiv_of(x, equiv_subtrees, tau=tau, metric=metric) or x.label() for x in group_ent_trees})
        k = len({x.label() for x in group_ent_trees})
        with tqdm(desc='k-groups', total=k, leave=False) as pbar:
            while k > 1:
                pbar.update(k)

                nb_combinations = math.comb(len(group_ent_trees), k)
                k_groups = combinations(group_ent_trees, k)

                if not k_groups:
                    break

                # Get k-subgroup with maximum support
                k_groups_support = list(filter(lambda x: x[0] is not None, TASK_POOL(
                    delayed(_merge_groups_inner)(
                        subtree, combined_groups, equiv_subtrees=equiv_subtrees, tau=tau, min_support=min_support,
                        metric=metric
                    )
                    for combined_groups in
                    tqdm(k_groups, leave=False, total=nb_combinations, desc=f'{subtree.label()} n={len(group_ent_trees)} k={k}')
                )))

                # Compute max merge
                if k_groups_support:
                    max_subtree, _ = max(
                        k_groups_support,
                        key=lambda x: x[1],  # if x[1] >= group_support else float('-inf'),
                        default=(None, 0)
                    )

                else:
                    max_subtree = None

                # If no k-group found, we reduce group size
                if max_subtree is None:
                    k -= 1
                    continue

                # A group is found, we need to add the new subgroup tree
                print(
                    f'merge_groups {subtree.treeposition()} k={k}:\t{[x.label() for x in subtree]} -> {[x.label() for x in max_subtree]}',
                    file=trace
                )
                trace.flush()
                simplified = True

                # Replace subtree with the newly constructed one
                parent.pop(parent_idx)
                parent.insert(parent_idx, ParentedTree.convert(Tree.convert(max_subtree)))
                update_cache(parent[parent_idx])

                # Remove used entity trees and start over
                entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), max_subtree))

                # Keep searching k-group (reduce k if the number of remaining entities is lower than current k)
                k = min(len(entity_trees), k)

    return simplified


def find_relationship(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC,
                      naming_only: bool = False) -> bool:
    simplified = False

    for subtree in tqdm(list(t.subtrees(lambda x: len(x) == 2 and not has_type(x) and x != t.root() and x.parent() != t.root())), desc='find relations', leave=False):
        group = None
        collection = None

        # Group <-> Group
        if has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.GROUP) and \
                subtree[0].label().name != subtree[1].label().name:
            label = sorted([subtree[0].label().name, subtree[1].label().name])
            subtree.set_label(NodeLabel(NodeType.REL, f'{label[0]} <-> {label[1]}'))
            continue

        if naming_only:
            continue

        # Collection <-> Group
        elif has_type(subtree[0], NodeType.COLL) and has_type(subtree[1], NodeType.GROUP):
            collection = subtree[0]
            group = subtree[1]

        # Group <-> Collection
        elif has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.COLL):
            group = subtree[0]
            collection = subtree[1]

        if group and collection:
            simplified = True
            print(
                f'find_relationship {subtree.treeposition()}:\tgroup={group.label()} coll={collection.label()}',
                file=trace
            )
            trace.flush()
            for coll_group in collection:
                label = sorted([group.label().name, coll_group.label().name])
                rel_tree = Tree(
                    NodeLabel(NodeType.REL, f'{label[0]} <-> {label[1]}'),
                    children=[group, coll_group]
                )
                ins_elem(subtree, rel_tree, 0)

            del_elem(subtree, group.parent_index())
            del_elem(subtree, collection.parent_index())

    return simplified


def find_collections(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC,
                     naming_only: bool = False) -> bool:
    simplified = False

    for subtree in tqdm(list(reversed(list(t.subtrees()))), desc='find collections', leave=False):

        # Make collection of group / rels or merge collections
        for coll_tree_set in sorted(
                filter(
                    lambda x: len(x) > 1,
                    (
                            sorted(equiv_set, key=lambda x: x.parent_index())
                            for _, equiv_set in groupby(
                        sorted(
                            filter(
                                lambda x: isinstance(x, Tree) and has_type(x, {NodeType.GROUP, NodeType.REL, NodeType.COLL}),
                                subtree
                            ),
                            key=lambda x: x.label().name
                        ),
                        key=lambda x: x.label().name
                    )
                    )
                ),
                key=lambda x: x[0].parent_index()
        ):
            # Add an intermediate collection node. If it is a collection of collection we merge them into one
            coll_elements = []
            for tree in coll_tree_set:
                if has_type(tree, NodeType.COLL):
                    simplified = True
                    coll_elements.extend(tree)
                else:
                    coll_elements.append(tree)

            label = NodeLabel(NodeType.COLL, coll_tree_set[0].label().name)
            coll_tree = Tree(label, children=[Tree.convert(tree) for tree in coll_elements])

            if naming_only:
                continue

            if len(subtree) == len(coll_tree_set) and subtree is not t.root():
                index = subtree.parent_index()
                subtree = subtree.parent()
                subtree.pop(index)

            else:
                simplified = True
                print(
                    f'find_collections {subtree.treeposition()}:\t{[x.label() for x in coll_tree_set]}',
                    file=trace
                )
                trace.flush()
                index = coll_tree_set[0].parent_index()
                for tree in coll_tree_set:
                    subtree.pop(tree.parent_index())

            ins_elem(subtree, coll_tree, index)

    return simplified

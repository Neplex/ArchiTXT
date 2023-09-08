from collections import Counter
from collections.abc import Callable
from itertools import combinations, groupby

from tqdm import tqdm

from .model import NodeType, NodeLabel
from .similarity import METRIC_FUNC, TREE_CLUSTER, get_equiv_of
from .tree import *

__all__ = [
    'OPERATION',
    'find_groups', 'find_subgroups', 'merge_groups',
    'find_relationship', 'find_collections',
    'reduce_bottom', 'reduce_top',
]

OPERATION = Callable[[ParentedTree, TREE_CLUSTER, float, int, METRIC_FUNC], bool]


def reduce_bottom(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    return any([
        reduce(subtree.parent(), subtree.parent_index(), list(NodeType))
        for subtree in t.subtrees(lambda x: any(has_type(child, NodeType.ENT) for child in x))
    ])


def reduce_top(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    return any([
        reduce(subtree.parent(), subtree.parent_index(), list(NodeType))
        for subtree in t.subtrees(lambda x: x.height() == (t.height() - 1))
    ])


def find_groups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC):
    frequent_subtrees = filter(lambda x: len(x) >= min_support, equiv_subtrees)
    frequent_subtrees = list(sorted(frequent_subtrees, key=lambda x: sum(y.depth() for y in x) / len(x)))

    k = 0
    with tqdm(leave=False) as pb:
        while frequent_subtrees:
            min_group = frequent_subtrees.pop(0)

            # Skip frequent subtrees containing or contain in groups
            if any(has_type(x, NodeType.GROUP) or has_type(subtree, NodeType.GROUP) for subtree in min_group for x in subtree):
                continue

            for subtree in min_group:
                entity_trees = [Tree.convert(ent_tree) for ent_tree in subtree.subtrees(lambda x: has_type(x, NodeType.ENT))]
                group_tree = Tree(NodeLabel(NodeType.GROUP, k), children=entity_trees)

                group_parent = subtree.parent()
                group_pos = subtree.parent_index()
                group_parent.pop(group_pos)
                ins_elem(group_parent, group_tree, group_pos)

            k += 1
            pb.update(1)


def find_subgroups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False
    equiv_subtrees = set(filter(lambda x: len(x) >= min_support, equiv_subtrees))

    for subtree in tqdm(reversed(list(t.subtrees(
            lambda x: not isinstance(x, str) and x != t.root() and not has_type(x)
    ))), desc='find subgroups', leave=False):
        group_support = len(get_equiv_of(subtree, equiv_subtrees, tau=tau, metric=metric))
        entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), subtree))
        subtree_pos = subtree.treeposition()
        parent_idx = subtree.parent_index()
        parent = subtree.parent()

        k = len(entity_trees) - 1
        while k > 1:
            max_support = 0
            max_subtree = None

            # Get k-subgroup with maximum support
            for sub_group in tqdm(sorted(combinations(entity_trees, k), key=lambda x: x[-1].parent_index() - x[0].parent_index()),
                                  leave=False, desc=f'{subtree.label()} k={k}'):
                if max(Counter(x.label() for x in sub_group).values()) > 1:
                    continue

                # Copy the tree
                new_tree = t.copy(deep=True)
                new_subtree = new_tree[subtree_pos]

                group_tree = Tree(NodeLabel(NodeType.GROUP), children=[Tree.convert(ent_tree) for ent_tree in sub_group])

                for ent_tree in sorted(sub_group, key=lambda x: x.parent_index(), reverse=True):
                    del_elem(new_subtree, ent_tree.parent_index())

                group_pos = min(ent_tree.parent_index() for ent_tree in sub_group)
                ins_elem(new_subtree, group_tree, group_pos)

                equiv_group = get_equiv_of(new_subtree[group_pos], equiv_subtrees, tau=tau, metric=metric)
                if (support := len(equiv_group)) > min_support and support > max_support and support > group_support:
                    if has_type(equiv_group[0], NodeType.GROUP):
                        new_subtree[group_pos].set_label(equiv_group[0].label())
                    max_support = support
                    max_subtree = new_subtree

            # If no k-group found, we reduce group size
            if not max_subtree:
                k -= 1
                continue

            # A group is found, we need to add the new subgroup tree
            simplified = True

            # Replace subtree with the newly constructed one
            parent.pop(parent_idx)
            parent.insert(parent_idx, ParentedTree.convert(Tree.convert(max_subtree)))

            # Remove used entity trees and start over
            entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), max_subtree))
            k = min(len(entity_trees), k)  # Keep searching k-group (reduce k if the number of remaining entities is lower than current k)

    return simplified


def merge_groups(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False

    for subtree in tqdm(reversed(list(t.subtrees(
            lambda x: not isinstance(x, str) and x != t.root() and not has_type(x)
    ))), desc='merge groups', leave=False):
        group_support = len(get_equiv_of(subtree, equiv_subtrees, tau=tau, metric=metric))
        group_ent_trees = list(filter(lambda x: has_type(x, (NodeType.GROUP, NodeType.ENT)), subtree))
        subtree_pos = subtree.treeposition()
        parent = subtree.parent()
        parent_idx = subtree.parent_index()

        k = len(group_ent_trees)
        while k > 1:
            max_support = 0
            max_subtree = None

            # Get k-subgroup with maximum support
            for combined_groups in tqdm(sorted(combinations(group_ent_trees, k), key=lambda x: x[-1].parent_index() - x[0].parent_index()),
                                        leave=False, desc=f'{subtree.label()} k={k}'):
                sub_group = []
                for group_ent in combined_groups:
                    if has_type(group_ent, NodeType.GROUP):
                        sub_group.extend(group_ent)
                    else:
                        sub_group.append(group_ent)

                if max(Counter(x.label() for x in sub_group).values()) > 1:
                    continue

                # Copy the tree
                new_tree = t.copy(deep=True)
                new_subtree = new_tree[subtree_pos]

                # Create new tree version
                group_tree = Tree(NodeLabel(NodeType.GROUP), children=[Tree.convert(ent_tree) for ent_tree in sub_group])
                group_pos = min(group_ent.parent_index() for group_ent in combined_groups)

                for group_ent in sorted(combined_groups, key=lambda x: x.parent_index(), reverse=True):
                    new_subtree.pop(group_ent.parent_index())
                ins_elem(new_subtree, group_tree, group_pos)

                # Compute equivalent class
                equiv_group = get_equiv_of(new_subtree[group_pos], equiv_subtrees, tau=tau, metric=metric)
                if (support := len(equiv_group)) > min_support and support > max_support and support >= group_support:
                    if has_type(equiv_group[0], NodeType.GROUP):
                        new_subtree[group_pos].set_label(equiv_group[0].label())
                    max_support = support
                    max_subtree = new_subtree

            # If no k-group found, we reduce group size
            if not max_subtree:
                k -= 1
                continue

            # A group is found, we need to add the new subgroup tree
            simplified = True

            # Replace subtree with the newly constructed one
            parent.pop(parent_idx)
            parent.insert(parent_idx, ParentedTree.convert(Tree.convert(max_subtree)))

            # Remove used entity trees and start over
            entity_trees = list(filter(lambda x: has_type(x, NodeType.ENT), max_subtree))
            k = min(len(entity_trees), k)  # Keep searching k-group (reduce k if the number of remaining entities is lower than current k)

    return simplified


def find_relationship(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False

    for subtree in tqdm(t.subtrees(lambda x: len(x) == 2 and not has_type(x) and x != t.root()), desc='find relations', leave=False):
        group = None
        collection = None

        # Group <-> Group
        if has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.GROUP):
            label = sorted([subtree[0].label().name, subtree[1].label().name])
            subtree.set_label(NodeLabel(NodeType.REL, f'{label[0]} <-> {label[1]}'))
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


def find_collections(t: ParentedTree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC) -> bool:
    simplified = False

    for subtree in tqdm(reversed(list(t.subtrees())), desc='find collections', leave=False):

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

            if len(subtree) == len(coll_tree_set) and subtree is not t.root():
                index = subtree.parent_index()
                subtree = subtree.parent()
                subtree.pop(index)

            else:
                simplified = True
                index = coll_tree_set[0].parent_index()
                for tree in coll_tree_set:
                    subtree.pop(tree.parent_index())

            ins_elem(subtree, coll_tree, index)

    return simplified

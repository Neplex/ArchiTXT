from collections.abc import Callable
from copy import deepcopy
from itertools import combinations, groupby

import mlflow
import more_itertools
from mlflow.entities import SpanEvent

from architxt.model import NodeLabel, NodeType
from architxt.similarity import METRIC_FUNC, TREE_CLUSTER, get_equiv_of
from architxt.tree import Tree, has_type

__all__ = [
    'OPERATION',
    'find_collections',
    'find_groups',
    'find_relations',
    'find_subgroups',
    'merge_groups',
    'reduce_bottom',
    'reduce_top',
]

OPERATION = Callable[[Tree, TREE_CLUSTER, float, int, METRIC_FUNC], tuple[Tree, bool]]


def reduce_bottom(
    tree: Tree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[Tree, bool]:
    """
    Reduces the unlabelled nodes of a tree at the bottom-level

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.

    :param tree: The tree to perform the reduction on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    reduced = False

    # Iterate through subtrees in reverse order to ensure bottom-up processing
    for subtree in tree.subtrees(lambda x: not has_type(x) and x.parent() and x.has_entity_child()):
        parent = subtree.parent()
        position = subtree.treeposition()
        label = subtree.label()
        old_labels = tuple(child.label() for child in parent)

        # Convert subtree's children into independent nodes
        new_children = [deepcopy(child) for child in subtree]

        # Put children in the parent at the original subtree position
        parent_pos = subtree.parent_index()
        parent[parent_pos:parent_pos] = reversed(new_children)

        new_labels = tuple(child.label() for child in parent)
        if span := mlflow.get_current_active_span():
            span.add_event(
                SpanEvent(
                    'reduce_bottom',
                    attributes={
                        'label': str(label),
                        'position': position,
                        'labels.old': old_labels,
                        'labels.new': new_labels,
                    },
                )
            )

        reduced = True

    return tree, reduced


def reduce_top(
    tree: Tree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[Tree, bool]:
    """
    Reduces the unlabelled nodes of a tree at the top-level

    This function identifies subtrees that do not have a specific type but contain children of type `ENT`.
    It then repositions these subtrees' children directly under their parent nodes, effectively "flattening"
    the tree structure at this level.

    :param tree: The tree to perform the reduction on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    reduced = False

    for subtree in tree.subtrees(lambda x: not has_type(x) and x.parent()):
        parent = subtree.parent()
        position = subtree.treeposition()
        label = subtree.label()
        old_labels = tuple(child.label() for child in parent)

        # Convert subtree's children into independent nodes
        new_children = [deepcopy(child) for child in subtree]

        # Put children in the parent at the original subtree position
        parent_pos = subtree.parent_index()
        parent[parent_pos:parent_pos] = reversed(new_children)

        new_labels = tuple(child.label() for child in parent)
        if span := mlflow.get_current_active_span():
            span.add_event(
                SpanEvent(
                    'reduce_top',
                    attributes={
                        'label': str(label),
                        'position': position,
                        'labels.old': old_labels,
                        'labels.new': new_labels,
                    },
                )
            )

        reduced = True

    return tree, reduced


def _create_group(subtree: Tree, group_index: int) -> None:
    """
    Creates a group node from a subtree and inserts it into its parent node.

    :param subtree: The subtree to convert into a group.
    :param group_index: The index to use for naming the group.
    """
    label = NodeLabel(NodeType.GROUP, str(group_index))
    subtree.set_label(label)

    new_children = [deepcopy(entity) for entity in subtree.entities()]
    subtree.clear()
    subtree.extend(new_children)


def find_groups(
    tree: Tree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[Tree, bool]:
    """
    Finds and creates groups in the tree based on equivalent subtrees.

    :param tree: The tree to perform the reduction on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    frequent_clusters = filter(lambda cluster: len(cluster) > min_support, equiv_subtrees)
    frequent_clusters = sorted(
        frequent_clusters,
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
                any(has_type(node, NodeType.GROUP) for node in subtree)
                or (subtree.parent() and has_type(subtree.parent(), NodeType.GROUP))
                or len(subtree) < 2
            ):
                continue

            _create_group(subtree, group_index)
            group_created = True

        # if group_created:
        group_labels = tuple(sorted({label for subtree in subtree_cluster for label in subtree.entity_labels()}))
        if span := mlflow.get_current_active_span():
            span.add_event(
                SpanEvent(
                    'create_group',
                    attributes={
                        'group': group_index,
                        'num_instance': len(subtree_cluster),
                        'labels': group_labels,
                    },
                )
            )

    return tree, group_created


def _find_subgroups_inner(
    subtree: Tree,
    sub_group: tuple[Tree, ...],
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
) -> tuple[Tree, int] | None:
    """
    Attempts to add a new subtree by creating a new `GROUP` for a given `sub_group` of entities, and evaluates its
    support within the `equiv_subtrees` equivalence class.

    :param subtree: The tree structure within which a potential subgroup will be created.
    :param sub_group: A tuple of `Tree` entities to be grouped into a new `GROUP` node.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: A tuple containing the modified subtree and its support count if the modified subtree
             meets the minimum support threshold; otherwise, `None`.
    """
    # Create a copy of the tree we worked on
    new_tree = deepcopy(subtree.root())
    new_subtree = new_tree[subtree.treeposition()]

    # Create the new GROUP node
    group_tree = Tree(NodeLabel(NodeType.GROUP), children=[deepcopy(ent_tree) for ent_tree in sub_group])

    # Removed used entity trees from the subtree
    for ent_tree in sorted(sub_group, key=lambda x: x.parent_index(), reverse=True):
        new_subtree.pop(ent_tree.parent_index())

    # Insert the GROUP node at the position of the earliest entity in sub_group
    insertion_index = min(ent_tree.parent_index() for ent_tree in sub_group)
    new_subtree.insert(insertion_index, group_tree)

    # If the subtree was a Group, it is not valid anymore and should not keep its label
    if has_type(subtree, NodeType.GROUP):
        new_subtree.set_label('')

    # We compute the support of the new subtree's. It is a valid subgroup if its support exceeds the given threshold.
    equiv = get_equiv_of(new_subtree[insertion_index], equiv_subtrees, tau=tau, metric=metric)
    support = len(equiv)

    if support > min_support:
        new_subtree[insertion_index].set_label(equiv[0].label())
        return new_subtree, support

    return None


def find_subgroups(
    tree: Tree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[Tree, bool]:
    """
    Identifies and create subgroup of entities for each subtree if the support of the newly created subgroup is greater
    that the support of the subtree.

    :param tree: The tree to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    simplified = False

    for subtree in sorted(
        tree.subtrees(
            lambda sub: not has_type(sub, {NodeType.ENT, NodeType.REL, NodeType.COLL})
            and all(has_type(child, NodeType.ENT) for child in sub)
        ),
        key=lambda sub: sub.height(),
    ):
        # Calculate initial support for the subtree
        group_support = len(get_equiv_of(subtree, equiv_subtrees, tau=tau, metric=metric))
        entity_trees = tuple(filter(lambda child: has_type(child, NodeType.ENT), subtree))
        parent = subtree.parent()
        parent_idx = subtree.parent_index()

        k = min(len(entity_trees), len(subtree) - 1)

        # Recursively creating k-sized groups, decreasing k if necessary
        while k > 1:
            # Generate k-sized combinations of entity trees and keep the one with maximum support
            k_groups = combinations(entity_trees, k)
            k_groups_support = (
                _find_subgroups_inner(
                    subtree,
                    sub_group,
                    equiv_subtrees=equiv_subtrees,
                    tau=tau,
                    min_support=max(group_support, min_support - 1),
                    metric=metric,
                )
                for sub_group in k_groups
                if more_itertools.all_unique(ent.label() for ent in sub_group)
            )

            # Get the subgroup with the maximum support
            max_subtree: Tree | None
            max_subtree, max_support = max(
                filter(lambda result: result is not None, k_groups_support),
                key=lambda result: result[1],
                default=(None, None),
            )

            # No suitable k-group found; decrease k and try again
            if not max_subtree:
                k -= 1
                continue

            # Successfully found a valid k-group, mark the tree as simplified
            simplified = True
            if span := mlflow.get_current_active_span():
                span.add_event(
                    SpanEvent(
                        'find_subgroup',
                        attributes={
                            'num_instance': max_support,
                            'labels': [str(ent.label()) for ent in max_subtree],
                        },
                    )
                )

            # Replace subtree with the newly constructed one
            if parent:
                subtree = parent[parent_idx] = deepcopy(max_subtree)

            else:
                subtree.clear()
                subtree.extend(deepcopy(max_subtree[:]))

            # Update entity trees and reset k for remaining entities
            entity_trees = tuple(filter(lambda child: has_type(child, NodeType.ENT), subtree))
            k = min(len(entity_trees), k)

    return tree, simplified


def _merge_groups_inner(
    subtree: Tree,
    combined_groups: tuple[Tree, ...],
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
) -> tuple[Tree, int] | None:
    """
    Attempts to merge specified `GROUP` and `ENT` nodes within a subtree by replacing them with a single `GROUP` node,
    given that it meets minimum support and subtree similarity requirements.

    :param subtree:
    :param combined_groups:
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: A tuple containing the modified subtree and its support count if the modified subtree
             meets the minimum support threshold; otherwise, `None`.
    """
    sub_group = []
    max_sub_group_support = 0
    group_count = 0

    for group_entity in combined_groups:
        # Directly append single `ENT` nodes
        if has_type(group_entity, NodeType.ENT):
            sub_group.append(group_entity)

        # Process `GROUP` nodes, treating single-element groups as entities
        elif has_type(group_entity, NodeType.GROUP):
            if len(group_entity) == 1:  # Group of sizes 1 are treated as entities
                sub_group.append(group_entity[0])

            else:
                group_count += 1
                group_support = len(get_equiv_of(group_entity, equiv_subtrees, tau=tau, metric=metric))
                max_sub_group_support = max(max_sub_group_support, group_support)
                sub_group.extend(group_entity.entities())

    # Skip if invalid conditions are met: duplicates entities, empty groups, or no valid subgroups
    if not sub_group or group_count == 0 or not more_itertools.all_unique(ent.label() for ent in sub_group):
        return None

    # Copy the tree
    new_tree = deepcopy(subtree.root())
    new_subtree = new_tree[subtree.treeposition()]

    # Create new `GROUP` node with selected entities
    group_tree = Tree(NodeLabel(NodeType.GROUP), children=[deepcopy(ent) for ent in sub_group])

    # Removed used entity trees from the subtree
    for group_ent in sorted(combined_groups, key=lambda x: x.parent_index(), reverse=True):
        new_subtree.pop(group_ent.parent_index())

    # Insert the newly created `GROUP` node at the appropriate position
    group_position = min(group_entity.parent_index() for group_entity in combined_groups)
    new_subtree.insert(group_position, group_tree)

    # Compute support for the newly formed group
    equiv = get_equiv_of(new_subtree[group_position], equiv_subtrees, tau=tau, metric=metric)
    support = len(equiv)

    # Return the modified subtree and its support counts if support exceeds the threshold
    if support > min_support and support >= max_sub_group_support:
        new_subtree[group_position].set_label(equiv[0].label())
        return new_subtree, support

    return None


def merge_groups(
    tree: Tree, equiv_subtrees: TREE_CLUSTER, tau: float, min_support: int, metric: METRIC_FUNC
) -> tuple[Tree, bool]:
    """
    Attempts to add `ENT` to existing `GROUP` within a tree by forming new `GROUP` nodes that does not reduce the
    support of the given group.

    :param tree: The tree to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    simplified = False

    for subtree in sorted(
        tree.subtrees(lambda x: not has_type(x) and any(has_type(y, NodeType.GROUP) for y in x)),
        key=lambda x: x.height(),
    ):
        # Identify `GROUP` and `ENT` nodes in the subtree that could be merged
        group_ent_trees = tuple(filter(lambda x: has_type(x, {NodeType.GROUP, NodeType.ENT}), subtree))
        parent = subtree.parent()
        parent_idx = subtree.parent_index()

        k = len({x.label() for x in group_ent_trees})

        # Recursively creating k-sized groups, decreasing k if necessary
        while k > 1:
            # Get k-subgroup with maximum support
            k_groups = combinations(group_ent_trees, k)
            k_groups_support = (
                _merge_groups_inner(
                    subtree,
                    combined_groups,
                    equiv_subtrees=equiv_subtrees,
                    tau=tau,
                    min_support=min_support,
                    metric=metric,
                )
                for combined_groups in k_groups
            )

            # Identify the best possible merge based on maximum support
            max_subtree: Tree | None
            max_subtree, max_support = max(
                filter(lambda x: x is not None, k_groups_support),
                key=lambda x: x[1],
                default=(None, None),
            )

            # If no valid k-sized group was found, reduce k and continue
            if max_subtree is None:
                k -= 1
                continue

            # A group is found, we need to add the new subgroup tree
            simplified = True
            if span := mlflow.get_current_active_span():
                span.add_event(
                    SpanEvent(
                        'group_merged',
                        attributes={
                            'num_instance': max_support,
                            'labels': [str(ent.label()) for ent in max_subtree],
                        },
                    )
                )

            # Replace subtree with the newly constructed one
            if parent:
                subtree = parent[parent_idx] = deepcopy(max_subtree)

            else:
                subtree.clear()
                subtree.extend(deepcopy(max_subtree[:]))

            # Update entity trees and reset k for remaining entities
            group_ent_trees = tuple(filter(lambda child: has_type(child, {NodeType.GROUP, NodeType.ENT}), subtree))
            k = min(len(group_ent_trees), k)

    return tree, simplified


def find_relations(
    tree: Tree,
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    *,
    naming_only: bool = False,
) -> tuple[Tree, bool]:
    """
    Identifies and establishes hierarchical relationships between `GROUP` nodes within a tree structure.

    The function scans for subtrees that contain at least two distinct elements.
    When a `GROUP` node is found to have a relationship with a collection, that relationship
    is distributed between the `GROUP` node itself and each member of the collection.

    This function can also apply naming-only transformations without structural modifications.

    :param tree: The tree to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param naming_only: If True, the operation only names valid relations without rewriting the tree.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    simplified = False

    # Traverse subtrees, starting with the deepest, containing exactly 2 children
    for subtree in sorted(
        tree.subtrees(
            lambda x: len(x) == 2 and not has_type(x) and any(has_type(y, {NodeType.GROUP, NodeType.COLL}) for y in x)
        ),
        key=lambda x: x.depth(),
        reverse=True,
    ):
        group = None
        collection = None

        # Group <-> Group
        if (
            has_type(subtree[0], NodeType.GROUP)
            and has_type(subtree[1], NodeType.GROUP)
            and subtree[0].label().name != subtree[1].label().name
        ):
            # Create and set relationship label
            label = sorted([subtree[0].label().name, subtree[1].label().name])
            subtree.set_label(NodeLabel(NodeType.REL, f'{label[0]}<->{label[1]}'))

            # Log relation creation in MLFlow, if active
            if span := mlflow.get_current_active_span():
                span.add_event(
                    SpanEvent(
                        'find_relation',
                        attributes={
                            'name': f'{label[0]}<->{label[1]}',
                        },
                    )
                )
            continue

        # If only naming relationships, skip further processing
        if naming_only:
            continue

        # Group <-> Collection
        if has_type(subtree[0], NodeType.GROUP) and has_type(subtree[1], NodeType.COLL):
            group, collection = subtree[0], subtree[1]

        elif has_type(subtree[0], NodeType.COLL) and has_type(subtree[1], NodeType.GROUP):
            collection, group = subtree[0], subtree[1]

        # If a valid Group-Collection pair is found, create relationships for each
        if group and collection:
            simplified = True

            # Create relationship nodes for each element in the collection
            for coll_group in collection:
                label1, label2 = sorted([group.label().name, coll_group.label().name])
                rel_label = NodeLabel(NodeType.REL, f'{label1}<->{label2}')
                rel_tree = Tree(rel_label, children=deepcopy([group, coll_group]))
                subtree.append(rel_tree)  # Add new relationship to subtree

                # Log relation creation in MLFlow, if active
                if span := mlflow.get_current_active_span():
                    span.add_event(
                        SpanEvent(
                            'find_relation',
                            attributes={
                                'name': rel_label.name,
                            },
                        )
                    )

            subtree.remove(group)
            subtree.remove(collection)

    return tree, simplified


def find_collections(
    tree: Tree,
    equiv_subtrees: TREE_CLUSTER,
    tau: float,
    min_support: int,
    metric: METRIC_FUNC,
    *,
    naming_only: bool = False,
) -> tuple[Tree, bool]:
    """
    Identifies and groups nodes into collections within a tree.

    This function can also apply naming-only transformations without structural modifications.

    :param tree: The tree to perform on.
    :param equiv_subtrees: The set of equivalent subtrees.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support of groups.
    :param metric: The metric function used to compute similarity between subtrees.
    :param naming_only: If True, the operation only names valid collections without rewriting the tree.

    :return: The modified tree and boolean indicating if the tree was reduced.
    """
    simplified = False

    for subtree in sorted(
        tree.subtrees(
            lambda x: not has_type(x) and any(has_type(y, {NodeType.GROUP, NodeType.REL, NodeType.COLL}) for y in x)
        ),
        key=lambda x: x.depth(),
        reverse=True,
    ):
        # Naming-only mode: apply labels without modifying tree structure
        if naming_only:
            if all(
                has_type(x, {NodeType.GROUP, NodeType.REL}) and x.label().name == subtree[0].label().name
                for x in subtree
            ):
                subtree.set_label(NodeLabel(NodeType.COLL, subtree[0].label().name))
            continue

        # Group nodes by shared label and organize them into collection sets for structural modification
        for coll_tree_set in sorted(
            filter(
                lambda x: len(x) > 1,
                (
                    sorted(equiv_set, key=lambda x: x.parent_index())
                    for _, equiv_set in groupby(
                        sorted(filter(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL, NodeType.COLL}), subtree)),
                        key=lambda x: x.label().name,
                    )
                ),
            ),
            key=lambda x: x[0].parent_index(),
        ):
            # Prepare a new collection of nodes (merging if some nodes are already collections)
            coll_elements = []
            for coll_tree in coll_tree_set:
                if has_type(coll_tree, NodeType.COLL):
                    simplified = True  # Mark the tree as modified
                    coll_elements.extend(coll_tree)  # Merge collection elements
                else:
                    coll_elements.append(coll_tree)

            # Prepare the collection node
            label = NodeLabel(NodeType.COLL, coll_tree_set[0].label().name)
            children = [deepcopy(tree) for tree in coll_elements]

            # Log the creation of a new collection in MLFlow, if active
            if span := mlflow.get_current_active_span():
                span.add_event(
                    SpanEvent(
                        'find_collection',
                        attributes={
                            'name': label.name,
                            'size': len(children),
                        },
                    )
                )

            # If the entire subtree is a single collection, update its label and structure directly
            if len(subtree) == len(coll_tree_set):
                subtree.set_label(label)
                subtree.clear()
                subtree.extend(children)

            else:
                simplified = True
                index = coll_tree_set[0].parent_index()

                # Remove nodes of the current collection set from the subtree
                for coll_tree in coll_tree_set:
                    subtree.pop(coll_tree.parent_index(), recursive=False)

                # Insert the new collection node at the appropriate index
                coll_tree = Tree(label, children=children)
                subtree.insert(index, coll_tree)

    return tree, simplified

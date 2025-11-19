from __future__ import annotations

import uuid
from itertools import combinations
from typing import TYPE_CHECKING, cast

import more_itertools

from architxt.tree import NodeLabel, NodeType, Tree, has_type, is_sub_tree

from .operation import Operation

if TYPE_CHECKING:
    from architxt.similarity import TREE_CLUSTER
    from architxt.tree import _TypedSubTree

__all__ = [
    'FindSubGroupsOperation',
    'MergeGroupsOperation',
]


class FindSubGroupsOperation(Operation):
    """
    Identifies and create subgroup of entities for each subtree.

    It creates a group only if the support of the newly created subgroup is greater than the support of the subtree.
    """

    def _create_and_evaluate_subgroup(
        self, subtree: Tree, sub_group: tuple[_TypedSubTree, ...], min_support: int, equiv_subtrees: TREE_CLUSTER
    ) -> tuple[Tree, int] | None:
        """
        Attempt to add a new subtree by creating a new `GROUP` for a given `sub_group` of entities.

        It also evaluates the new group support within the `equiv_subtrees` equivalence class.

        :param subtree: The tree structure within which a potential subgroup will be created.
        :param sub_group: A tuple of `Tree` entities to be grouped into a new `GROUP` node.
        :param min_support: The support needed for a subgroup to be considered valid.
        :param equiv_subtrees: The cluster of equivalent subtrees in the forest.

        :return: A tuple containing the modified subtree and its support count if the modified subtree
                 meets the minimum support threshold; otherwise, `None`.
        """
        new_subtree = subtree.copy()

        # Create a new GROUP node with the given entities from the sub_group.
        group_tree = Tree(NodeLabel(NodeType.GROUP), children=[ent.copy() for ent in sub_group])

        # Remove the used entities from the original subtree
        # and insert the new GROUP node at the earliest index of the sub_group.
        indices = sorted((ent.parent_index for ent in sub_group), reverse=True)
        for idx in indices:
            new_subtree.pop(idx)

        # Insert the GROUP node at the position of the earliest entity in sub_group
        insertion_index = min(indices)
        new_subtree.insert(insertion_index, group_tree)

        # Reset label if subtree becomes invalid as a group
        if has_type(subtree):
            new_subtree.label = f'UNDEF_{uuid.uuid4().hex}'

        # Compute support for the new subtree. It is a valid subgroup if its support exceeds the given threshold.
        if equiv_class_name := self.get_equiv_of(group_tree, equiv_subtrees=equiv_subtrees):
            support = len(equiv_subtrees.get(equiv_class_name, ()))

            if support >= min_support:
                group_tree.label = NodeLabel(NodeType.GROUP, equiv_class_name)
                return new_subtree, support

        return None

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:
        simplified = False

        # Generate candidate subtrees that do not include ENT, REL, or COLL nodes as their children.
        candidate_subtrees = sorted(
            tree.subtrees(lambda sub: all(has_type(child, NodeType.ENT) for child in sub)),
            key=lambda sub: sub.height,
        )

        for subtree in candidate_subtrees:
            # Compute initial support for the subtree
            if equiv_class_name := self.get_equiv_of(subtree, equiv_subtrees=equiv_subtrees):
                group_support = len(equiv_subtrees.get(equiv_class_name, ()))
            else:
                group_support = 0

            entity_trees = [child for child in subtree if has_type(child, NodeType.ENT)]
            entity_labels = {ent.label for ent in entity_trees}

            # To narrow down the search space, we focus on reducing the entity trees to consider.
            # We retain only those groups that appear in clusters with higher support than the actual subtree,
            # and where the entity set intersects with the current subtrees.
            # This allows us to reduce the set of entity labels to consider only those present in these selected groups.
            entity_groups = {
                tuple(sorted(x.label for x in subtree))
                for cluster in equiv_subtrees.values()
                if len(cluster) > group_support
                for subtree in cluster
                if entity_labels.intersection(x.label for x in subtree)
            }

            if not entity_groups:
                continue

            available_labels = {label for group in entity_groups for label in group}

            # In addition to limiting the search to a subset of entity labels,
            # we can also restrict the size of subgroups to consider.
            # This helps prevent combinatorial explosion by avoiding the evaluation of excessively large groups.
            #
            # In one hane, we know that subgroups should be smaller than the actual subtree.
            # On the other hand, similarity is unlikely when groups differ significantly in size.
            # We can limit subgroup size to the largest group in the selected clusters
            # that contain a subset of the available entity labels within the subtree.
            # Larger groups could then be constructed by the merge_group operation.
            entity_trees = [entity for entity in entity_trees if entity.label in available_labels]
            entity_labels = {ent.label for ent in entity_trees}

            k = min(
                len(entity_trees),
                len(subtree) - 1,
                max(
                    (len(ent_group) for ent_group in entity_groups if entity_labels.issuperset(ent_group)),
                    default=len(entity_trees),
                ),
            )

            # Recursively explore k-sized combinations of entity trees and select the one with the maximum support,
            # decreasing k if necessary
            while k > 1:
                # Evaluate all k-groups
                evaluated_groups = (
                    new_sub_group
                    for sub_group in combinations(entity_trees, k)
                    if more_itertools.all_unique(ent.label for ent in sub_group)
                    and (
                        new_sub_group := self._create_and_evaluate_subgroup(
                            subtree,
                            sub_group,
                            min_support=max(group_support + 1, self.min_support),
                            equiv_subtrees=equiv_subtrees,
                        )
                    )
                )

                # Select the subgroup with maximum support
                max_subtree, max_support = max(evaluated_groups, key=lambda x: x[1], default=(None, None))

                # If no suitable k-group found; decrease k and try again
                if max_subtree is None:
                    k -= 1
                    continue

                # Successfully found a valid k-group, mark the tree as simplified
                simplified = True
                self._log_to_mlflow({'num_instance': max_support, 'labels': [str(ent.label) for ent in max_subtree]})

                # Replace the subtree with the newly constructed one
                if is_sub_tree(subtree):
                    subtree.parent[subtree.parent_index] = max_subtree
                    subtree = max_subtree
                else:
                    subtree[:] = [child.detach() for child in max_subtree[:]]

                # Reset entity trees and k
                entity_trees = [child for child in subtree if has_type(child, NodeType.ENT)]
                k = min(len(entity_trees), k)

        return simplified


class MergeGroupsOperation(Operation):
    """
    Attempt to add `ENT` to an existing ` GROUP ` within a tree.

    It tries to form a new `GROUP` node that does not reduce the support of the given group.
    """

    def _merge_groups_inner(
        self,
        subtree: Tree,
        combined_groups: tuple[_TypedSubTree, ...],
        equiv_subtrees: TREE_CLUSTER,
    ) -> tuple[Tree, int] | None:
        """
        Attempt to merge specified `GROUP` and `ENT` nodes within a subtree.

        It tries to replace them with a single `GROUP` node,
        given that it meets minimum support and subtree similarity requirements.

        :param subtree: The subtree to be modified during the merging process.
        :param combined_groups: A tuple containing subtrees or groups of subtrees to combine.
        :param equiv_subtrees: The cluster of equivalent subtrees in the forest.

        :return: A tuple containing the modified subtree and its support count if the modified subtree
                 meets the minimum support threshold; otherwise, `None`.
        """
        sub_group = []
        max_sub_group_support = 1
        group_count = 0

        for group_entity in combined_groups:
            # Directly append single `ENT` nodes
            if has_type(group_entity, NodeType.ENT):
                sub_group.append(group_entity)

            # Process `GROUP` nodes, treating single-element groups as entities
            elif has_type(group_entity, NodeType.GROUP):
                group_count += 1
                if equiv_class_name := self.get_equiv_of(group_entity, equiv_subtrees=equiv_subtrees):
                    group_support = len(equiv_subtrees.get(equiv_class_name, ()))
                    max_sub_group_support = max(max_sub_group_support, group_support)
                sub_group.extend(group_entity.entities())

        # Skip if invalid conditions are met: duplicates entities, empty groups, or no valid subgroups
        if not sub_group or group_count == 0 or not more_itertools.all_unique(ent.label for ent in sub_group):
            return None

        # Copy the tree
        new_tree = subtree.root.copy()
        new_subtree = cast('Tree', new_tree[subtree.position])  # new_subtree is a Tree as it is a copy of subtree

        # Create new `GROUP` node with selected entities
        group_tree = Tree(NodeLabel(NodeType.GROUP), children=[ent.copy() for ent in sub_group])

        # Removed used entity trees from the subtree
        for group_ent in sorted(combined_groups, key=lambda x: x.parent_index, reverse=True):
            new_subtree.pop(group_ent.parent_index, recursive=False)

        # Insert the newly created `GROUP` node at the appropriate position
        group_position = min(group_entity.parent_index for group_entity in combined_groups)
        new_subtree.insert(group_position, group_tree)

        # Compute support for the newly formed group
        if equiv_class_name := self.get_equiv_of(group_tree, equiv_subtrees=equiv_subtrees):
            support = len(equiv_subtrees.get(equiv_class_name, ()))

            # Return the modified subtree and its support counts if support exceeds the threshold
            if support >= max_sub_group_support:
                group_tree.label = NodeLabel(NodeType.GROUP, equiv_class_name)
                return new_subtree.detach(), support

        return None

    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:
        simplified = False

        for subtree in sorted(
            tree.subtrees(lambda x: not has_type(x) and any(has_type(y, NodeType.GROUP) for y in x)),
            key=lambda x: x.height,
        ):
            # Identify `GROUP` and `ENT` nodes in the subtree that could be merged
            group_ent_trees = tuple(filter(lambda x: has_type(x, {NodeType.GROUP, NodeType.ENT}), subtree))
            k = len({x.label for x in group_ent_trees})

            # Recursively creating k-sized groups, decreasing k if necessary
            while k > 1:
                # Get k-subgroup with maximum support
                k_groups = combinations(group_ent_trees, k)
                k_groups_support = (
                    self._merge_groups_inner(subtree, combined_groups, equiv_subtrees) for combined_groups in k_groups
                )

                # Identify the best possible merge based on maximum support
                max_subtree: Tree | None
                max_subtree, max_support = max(
                    filter(None, k_groups_support),
                    key=lambda x: x[1],
                    default=(None, 0),
                )

                # If no valid k-sized group was found, reduce k and continue
                if max_subtree is None:
                    k -= 1
                    continue

                # A group is found, we need to add the new subgroup tree
                simplified = True
                self._log_to_mlflow({'num_instance': max_support, 'labels': [str(ent.label) for ent in max_subtree]})

                # Replace the subtree with the newly constructed one
                if is_sub_tree(subtree):
                    subtree.parent[subtree.parent_index] = max_subtree
                    subtree = max_subtree
                else:
                    subtree[:] = [child.detach() for child in max_subtree[:]]

                # Update entity trees and reset k for remaining entities
                group_ent_trees = tuple(filter(lambda child: has_type(child, {NodeType.GROUP, NodeType.ENT}), subtree))
                k = min(len(group_ent_trees), k)

        return simplified

from copy import deepcopy

from tqdm import tqdm

from architxt.model import NodeLabel, NodeType
from architxt.tree import Forest, Tree


def simple_rewrite(forest: Forest, **kwargs) -> Forest:
    """
    Rewrites a given forest into a valid schema, treating each tree as a distinct group.
    Entities within a tree are grouped together, and duplicate entities are discarded.
    The function creates a unique group name for each distinct set of entities.

    :param forest: Input forest consisting of a list of Tree objects.
    :return: A new forest where each tree is restructured as a valid group. Already valid trees are kept as is.
    """
    new_forest: list[Tree] = []
    group_ids: dict[tuple[str, ...], str] = {}

    for tree in tqdm(forest):
        if not tree.has_unlabelled_nodes():
            new_forest.append(tree)
            continue

        entities = tree.entity_labels()
        group_key = tuple(sorted(entities))

        if group_key not in group_ids:
            group_ids[group_key] = str(len(group_ids) + 1)

        group_label = NodeLabel(NodeType.GROUP, group_ids[group_key])
        group_entities: list[Tree] = []

        for entity in tree.entities():
            if entity.label().name in entities:
                group_entities.append(deepcopy(entity))
                entities.remove(entity.label().name)

        group_tree = Tree(group_label, group_entities)
        new_forest.append(group_tree)

    return new_forest

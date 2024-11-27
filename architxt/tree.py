import contextlib
from collections import Counter
from collections.abc import Collection, Iterable
from copy import deepcopy
from functools import cache
from typing import Any, TypeAlias, overload

import pandas as pd
from nltk.grammar import Production
from nltk.tokenize.util import align_tokens
from nltk.tree import ParentedTree

from architxt.model import TREE_POS, Entity, NodeLabel, NodeType, Relation, TreeEntity, TreeRel

__all__ = [
    'Forest',
    'Tree',
    'enrich_tree',
    'fix_all_coord',
    'fix_conj',
    'fix_coord',
    'has_type',
    'ins_ent',
    'ins_rel',
    'reduce',
    'reduce_all',
    'unnest_ent',
]


class Tree(ParentedTree):
    slots = ('_parent', '_label')
    _parent: 'Tree | None'
    _label: NodeLabel | str

    def __init__(self, node: NodeLabel | str, children: Iterable['Tree | str'] | None = None):
        super().__init__(node, children)

        if isinstance(node, NodeLabel):
            return

        if '::' in self._label:
            node_type, _, name = self._label.partition('::')
            with contextlib.suppress(ValueError):
                self._label = NodeLabel(NodeType(node_type), name)

        else:
            with contextlib.suppress(ValueError):
                self._label = NodeLabel(NodeType(self._label))

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f'{type(self)}(len={len(self)})'

    def __reduce__(self):
        return type(self), (self._label, tuple(self))

    def depth(self) -> int:
        """
        Returns the depth of the tree.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.depth()
        1
        >>> t[0].depth()
        2
        >>> t[0, 0].depth()
        3
        """
        return len(self.treeposition()) + 1

    @cache
    def groups(self) -> set[str]:
        """
        Returns a set of group names found within the tree.

        :return: A set of unique group names within the tree.

        Example:
        >>> t = Tree.fromstring('(S (GROUP::A x) (GROUP::B y) (X (GROUP::C z)))')
        >>> sorted(t.groups())
        ['A', 'B', 'C']
        """
        result = set()

        for child in self:
            if has_type(child, NodeType.GROUP):
                result.add(child.label().name)

            if isinstance(child, Tree):
                result.update(child.groups())

        return result

    @cache
    def group_instances(self, group_name: str) -> pd.DataFrame:
        """
        Returns a DataFrame containing all instances of a specified group within the tree.

        Each row in the DataFrame represents an instance of the group, and each column represents an entity in that group, with the value being a concatenated string of that entity's leaves.

        :param group_name: The name of the group to search for.
        :return: A pandas DataFrame containing instances of the specified group.

        Example:
        >>> t = Tree.fromstring('(S (GROUP::A (ENT::person Alice) (ENT::fruit apple)) '
        ...                     '(GROUP::A (ENT::person Bob) (ENT::fruit banana)) '
        ...                     '(GROUP::B (ENT::person Charlie) (ENT::animal dog)))')
        >>> t.group_instances("A")
          person   fruit
        0  Alice   apple
        1    Bob  banana

        >>> t.group_instances("B")
            person animal
        0  Charlie    dog
        """
        dataframes = []
        records = []

        for child in self:
            if has_type(child, NodeType.GROUP) and child.label().name == group_name:
                records.append(
                    {
                        sub_child.label().name: ' '.join(sub_child.leaves())
                        for sub_child in child
                        if has_type(sub_child, NodeType.ENT)
                    }
                )

            if isinstance(child, Tree):
                dataframes.append(child.group_instances(group_name))

        return pd.concat([pd.DataFrame.from_records(records), *dataframes], ignore_index=True).drop_duplicates()

    @cache
    def entities(self) -> tuple['Tree', ...]:
        """
        Returns a tuple of subtrees that are entities.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> list(t.entities()) == [t[0, 0], t[0, 1], t[1, 0], t[1, 1]]
        True
        >>> del t[0]
        >>> list(t.entities()) == [t[0, 0], t[0, 1]]
        True
        """
        result = []
        for child in self:
            if has_type(child, NodeType.ENT):
                result.append(child)

            if isinstance(child, Tree):
                result.extend(child.entities())

        return tuple(result)

    @cache
    def entity_labels(self) -> set[str]:
        """
        Return a set of entity labels present in the tree.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> sorted(t.entity_labels())
        ['animal', 'fruit', 'person']
        >>> sorted(t[0].entity_labels())
        ['fruit', 'person']
        >>> del t[0]
        >>> sorted(t.entity_labels())
        ['animal', 'person']
        """
        return {node.label().name for node in self.entities()}

    @cache
    def entity_label_count(self) -> Counter[NodeLabel]:
        """
        Returns a Counter object that counts the labels of entity subtrees.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.entity_label_count()
        Counter({'person': 2, 'fruit': 1, 'animal': 1})
        """
        return Counter(ent.label().name for ent in self.entities())

    @cache
    def has_duplicate_entity(self) -> bool:
        """
        Checks if there are duplicate entity labels.

        Example:
        >>> from architxt.tree import Tree
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_duplicate_entity()
        True
        >>> t[0].has_duplicate_entity()
        False
        """
        return any(v > 1 for v in self.entity_label_count().values())

    @cache
    def has_entity_child(self) -> bool:
        """
        Checks if there is at least one entity as direct children.

        Example:
        >>> from architxt.tree import Tree
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_entity_child()
        False
        >>> t[0].has_entity_child()
        True
        """
        return any(has_type(child, NodeType.ENT) for child in self)

    def merge(self, tree: 'Tree') -> 'Tree':
        """
        Merge two trees into one.
        The root of both trees becomes one while maintaining the level of each subtree.
        """
        return type(self)('S', [*self, *tree])

    def __reset_cache(self):
        """Reset cached properties"""
        self.groups.cache_clear()
        self.group_instances.cache_clear()
        self.entities.cache_clear()
        self.entity_labels.cache_clear()
        self.entity_label_count.cache_clear()
        self.has_duplicate_entity.cache_clear()
        self.has_entity_child.cache_clear()

        # Remove cache recursively
        if parent := self.parent():
            parent.__reset_cache()

    @overload
    def __setitem__(self, pos: TREE_POS, subtree: 'Tree | str'): ...

    @overload
    def __setitem__(self, pos: slice, subtree: 'list[Tree | str]'): ...

    def __setitem__(self, pos: TREE_POS | slice, subtree: 'list[Tree | str] | Tree | str'):
        super().__setitem__(pos, subtree)
        self.__reset_cache()

    def __delitem__(self, pos: TREE_POS | slice) -> None:
        super().__delitem__(pos)
        self.__reset_cache()

    def set_label(self, label: NodeLabel | str) -> None:
        super().set_label(label)
        self.__reset_cache()

    def append(self, child: 'Tree | str') -> None:
        super().append(child)
        self.__reset_cache()

    def extend(self, children: 'Iterable[Tree | str]') -> None:
        super().extend(children)
        self.__reset_cache()

    def remove(self, child: 'Tree | str', recursive: bool = True) -> None:
        super().remove(child)
        self.__reset_cache()

        if recursive and len(self) == 0 and (parent := self._parent) is not None:
            parent.remove(self)

    def insert(self, pos: int, child: 'Tree | str') -> None:
        super().insert(pos, child)
        self.__reset_cache()

    def pop(self, pos: int = -1, recursive: bool = True) -> 'Tree | str':
        """
        Deletes an element from the treeat the specified position `pos`.
        If the parent tree becomes empty after the deletion, recursively deletes the parent node.

        :param pos: The position (index) of the element to delete in the tree.
        :param recursive: If an empty tree should be removed from the parent.
        :return: The element at the position. The function modifies the tree in place.

        Example:
        >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
        >>> print(t[(1, 1)].pformat(margin=255))
        (NP (NNS apples))
        >>> subtree = t[1, 1].pop(0)
        >>> print(t.pformat(margin=255))
        (S (NP Alice) (VP (VB like)))
        >>> subtree = t.pop(0)
        >>> print(t.pformat(margin=255))
        (S (VP (VB like)))
        >>> subtree = t[0].pop(0, recursive=False)
        >>> print(t.pformat(margin=255))
        (S (VP ))
        """
        child = super().pop(pos)
        self.__reset_cache()

        if recursive and len(self) == 0 and (parent := self._parent) is not None:
            parent.remove(self)

        return child


Forest: TypeAlias = Collection[Tree]


def has_type(t: Any, types: set[NodeType | str] | NodeType | str | None = None) -> bool:
    """
    Check if the given tree object has the specified type(s).

    :param t: The object to check type for (can be a Tree, Production, or NodeLabel).
    :param types: The types to check for (can be a set of strings, a string, or None).
    :return: True if the object has the specified type(s), False otherwise.

    Example:
    >>> tree = Tree.fromstring('(S (ENT Alice) (REL Bob))')
    >>> has_type(tree, NodeType.ENT)  # Check if the tree is of type 'S'
    False
    >>> has_type(tree[0], NodeType.ENT)
    True
    >>> has_type(tree[0], 'ENT')
    True
    >>> has_type(tree[1], NodeType.ENT)
    False
    >>> has_type(tree[1], {NodeType.ENT, NodeType.REL})
    True
    """
    assert t is not None

    # Normalize type input
    if types is None:
        types = set(NodeType)
    elif not isinstance(types, set):
        types = {types}

    types = {t.value if isinstance(t, NodeType) else str(t) for t in types}

    # Check for the type in the respective object
    if isinstance(t, NodeLabel):
        label = t
    elif isinstance(t, Tree):
        label = t.label()
    elif isinstance(t, Production):
        label = t.lhs().symbol()
    else:
        return False

    return isinstance(label, NodeLabel) and label.type.value in types


def reduce(tree: Tree, pos: int, types: set[str | NodeType] | None = None) -> bool:
    """
    Reduces a subtree within a tree at the specified position `pos`. The reduction occurs only
    if the subtree at `pos` has exactly one child, or if it does not match a specific set of node types.
    If the subtree can be reduced, its children are lifted into the parent node at `pos`.

    :param tree: The tree in which the reduction will take place.
    :param pos: The index of the subtree to attempt to reduce.
    :param types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: `True` if the subtree was reduced, `False` otherwise.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> reduce(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB like) (NNS apples)))
    >>> reduce(t, 0)
    True
    >>> print(t.pformat(margin=255))
    (S Alice (VP (VB like) (NNS apples)))
    """
    assert tree is not None

    # Check if the tree at the specified position can be reduced
    if (
        not isinstance(tree[pos], Tree)  # Ensure the subtree at `pos` is a Tree
        or (types and has_type(tree[pos], types))  # Check if it matches the specified types
        or (not types and len(tree[pos]) > 1)  # If no types, only reduce if it has one child
    ):
        return False

    # Replace the original subtree by its children into the parent at `pos`
    tree[pos : pos + 1] = [deepcopy(child) for child in tree[pos]]

    return True


def reduce_all(tree: Tree, skip_types: set[str | NodeType] | None = None) -> None:
    """
    Recursively attempts to reduce all eligible subtrees in a tree. The reduction process continues
    until no further reductions are possible. Subtrees can be skipped if their types are listed in `skip_types`.

    :param tree: The tree in which reductions will be attempted.
    :param skip_types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: None. The tree is modified in place.

    Example:
    >>> t = Tree.fromstring("(S (X (Y (Z (NP Alice)))) (VP (VB likes) (NP (NNS apples))))")
    >>> reduce_all(t)
    >>> print(t.pformat(margin=255))
    (S Alice (VP likes apples))
    """
    assert tree is not None

    reduced = True
    while reduced:
        reduced = False

        for subtree in tree.subtrees(lambda st: isinstance(st, Tree) and st.parent() is not None):
            if reduce(subtree.parent(), subtree.parent_index(), types=skip_types):
                reduced = True
                break


def fix_coord(tree: Tree, pos: int) -> bool:
    """
    Fixes the coordination structure in a tree at the specified position `pos`.
    This function modifies the tree to ensure that the conjunctions are structured correctly
    according to the grammar rules of coordination.

    :param tree: The tree in which coordination adjustments will be made.
    :param pos: The index of the subtree within the parent tree that contains the coordination to fix.
    :return: `True` if the coordination was successfully fixed, `False` otherwise.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_coord(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))
    """
    assert tree is not None

    coord = None

    # Identify the coordination subtree
    for child in tree[pos]:
        if (
            isinstance(child, Tree)
            and child.label() == 'COORD'
            and len(child) > 0
            and isinstance(child[0], Tree)
            and child[0].label() == 'CCONJ'
        ):
            coord = child
            break

    if coord is None:
        return False

    coord_index = coord.parent_index()

    # Create the left and right parts of the conjunction
    left = Tree(tree[pos].label(), children=[Tree.convert(child) for child in tree[pos][:coord_index]])
    right = [Tree.convert(child) for child in coord[1:]]  # Get all NPs after the conjunction

    # Create the conjunction tree
    conj = Tree('CONJ', children=[left, *right])  # CONJ should include the left NP and the conjuncts

    # Insert the new structure back into the tree
    # If children remain on the right of the coordination, we keep the existing level
    new_tree = (
        Tree(tree[pos].label(), children=[conj] + [Tree.convert(child) for child in remaining_children])
        if (remaining_children := tree[pos][coord_index + 1 :])
        else conj
    )

    # Replace the old subtree
    tree[pos] = new_tree

    return True


def fix_conj(tree: Tree, pos: int) -> bool:
    """
    Fixes conjunction structures in a tree at the specified position `pos`.
    If the node at `pos` is labeled 'CONJ', the function flattens any nested conjunctions
    by replacing the node with a new tree that combines its children.

    :param tree: The tree in which the conjunction structure will be fixed.
    :param pos: The index of the 'CONJ' node to be processed.

    :return: `True` if the conjunction structure was modified, `False` otherwise.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))")
    >>> fix_conj(t[1], 1)
    False
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (CONJ (NP (NNS oranges)) (NP (NNS bananas))))))")
    >>> fix_conj(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert tree is not None

    # Check if the specified position is valid and corresponds to a 'CONJ' node
    if not isinstance(tree[pos], Tree) or tree[pos].label() != 'CONJ':
        return False

    new_children: list[Tree | str] = []
    # Collect children, flattening nested conjunctions
    for child in tree[pos]:
        if isinstance(child, Tree) and child.label() == 'CONJ':
            new_children.extend(child)  # Extend with children of the nested CONJ
        else:
            new_children.append(child)  # Append non-CONJ children

    # If no changes were made, return False
    if len(new_children) <= len(tree[pos]):
        return False

    # Create a new tree for the flattened conjunction
    new_tree = Tree('CONJ', children=[Tree.convert(t) for t in new_children])

    # Replace the original 'CONJ' node with the new tree
    tree[pos] = new_tree

    return True


def fix_all_coord(tree: Tree) -> None:
    """
    Fixes all coordination structures in a tree.

    This function iteratively applies `fix_coord` and `fix_conj` to the tree
    until no further modifications can be made. It ensures that the tree adheres
    to the correct syntactic structure for coordination and conjunctions.

    :param tree: The tree in which coordination structures will be fixed.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_all_coord(t)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))

    >>> t2 = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges) (COORD (CCONJ and) (NP (NNS bananas))))))))")
    >>> fix_all_coord(t2)
    >>> print(t2.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert tree is not None

    # Fix coordination
    coord_fixed = True
    while coord_fixed:
        coord_fixed = False
        for pos in tree.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix coordination
            if fix_coord(tree[pos[:-1]], pos[-1]):
                coord_fixed = True
                break

    # Fix conjunctions
    conj_fixed = True
    while conj_fixed:
        conj_fixed = False
        for pos in tree.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix conjunctions
            if fix_conj(tree[pos[:-1]], pos[-1]):
                conj_fixed = True
                break


def ins_ent(tree: Tree, tree_ent: TreeEntity) -> Tree:
    """
    Inserts a tree entity into the appropriate position within a parented tree. The function modifies the tree
    structure to insert an entity at the correct level based on its positions and root position.

    :param tree: A tree representing the syntactic tree.
    :param tree_ent: A `TreeEntity` containing the entity name and its positions in the tree.
    :return: The updated subtree where the entity was inserted.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent1 = TreeEntity(name="person", positions=[(0, 0)])
    >>> tree_ent2 = TreeEntity(name="fruit", positions=[(1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent1)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (NP (NNS apples))))
    >>> ent_tree = ins_ent(t, tree_ent2)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (ENT::fruit apples)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (ENT::xxx like apples))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice apples) (VP (VB like)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like apples))
    >>> t_ent = TreeEntity(name="yyy", positions=[(0, 2)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like (ENT::yyy apples)))
    """
    assert tree is not None

    # Determine the insertion point based on the positions of the entity
    anchor_pos = tree_ent.root_pos
    anchor_pos_len = len(anchor_pos)
    child_pos = tree_ent.positions[0]

    if sum(child_pos[anchor_pos_len + 1 :]) > 0:
        # Entity has children; attach to the common parent at the first child index + 1
        entity_index = child_pos[anchor_pos_len] + 1

    elif (
        tree[tree_ent.root_pos].parent() is None
        or child_pos[anchor_pos_len] > 0
        or tree_ent.positions[-1][anchor_pos_len] < (len(tree[tree_ent.root_pos]) - 1)
    ):
        # Attach to common parent at the correct index
        entity_index = child_pos[anchor_pos_len]

    else:
        # Attach to the grandparent at the common parent index
        entity_index = tree_ent.root_pos[-1]
        anchor_pos = tree_ent.root_pos[:-1]

        # Adjust anchor position upwards if necessary
        while len(tree[anchor_pos]) == 1 and tree[anchor_pos].parent():
            entity_index = anchor_pos[-1]
            anchor_pos = anchor_pos[:-1]

    # Collect and delete children from the original positions
    children = []
    for child_position in reversed(tree_ent.positions):
        children.append(tree[child_position])
        tree[child_position[:-1]].pop(child_position[-1])

    # Create a new tree node for the entity and insert it into the tree
    new_tree = Tree(NodeLabel(NodeType.ENT, tree_ent.name), children=reversed(children))
    tree[anchor_pos].insert(entity_index, new_tree)

    # Return the modified subtree where the entity was inserted
    return tree[anchor_pos][entity_index]


def unnest_ent(tree: Tree, pos: int) -> None:
    """
    Un-nests an entity in a tree at the specified position `pos`.
    If the node at `pos` is labeled as an entity (ENT), the function converts
    the nested structure into a flat structure, creating a relationship (REL)
    between the entity and its nested entities.

    :param tree: The tree in which the entity will be un-nested.
    :param pos: The index of the 'ENT' node to be processed.

    Example:
    >>> t = Tree.fromstring('(S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))')
    >>> unnest_ent(t, 0)
    >>> print(t.pformat(margin=255))
    (S (REL (ENT::person Alice Bob Charlie) (nested (ENT::person Bob) (ENT::person Charlie))))
    """
    assert tree is not None

    # Check if the specified position corresponds to an 'ENT' node
    if not has_type(tree[pos], NodeType.ENT):
        return

    # Create the main entity tree and collect nested entities
    entity_tree = Tree(tree[pos].label(), children=tree[pos].leaves())

    # Collect nested entities and ensure they are only from the children of the current entity
    nested_entities = [deepcopy(child) for child in tree[pos] if has_type(child, NodeType.ENT)]
    nested_tree = Tree('nested', children=nested_entities)

    # Construct a new relationship tree with the entity and its nested entities
    new_tree = Tree(NodeLabel(NodeType.REL), children=[entity_tree, nested_tree]) if nested_entities else entity_tree

    # Replace the original entity node with the new structure
    tree[pos] = new_tree


def ins_rel(tree: Tree, tree_rel: TreeRel) -> None:
    assert tree is not None


def enrich_tree(tree: Tree, sentence: str, entities: list[Entity], relations: list[Relation]) -> None:
    """
    Enriches a syntactic tree (tree) by inserting entities and relationships, and removing unused subtrees.

    The function processes a list of entities and relations, inserting them into the tree, unnesting entities as needed,
    and finally deleting any subtrees that are not part of the enriched structure.

    :param tree: A tree representing the syntactic tree to enrich.
    :param sentence: The original sentence from which the tree is derived.
    :param entities: A list of `Entity` objects to be inserted into the tree.
    :param relations: A list of `Relation` objects representing the relationships between entities (currently not used).

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB likes) (NP (NNS apples) (CCONJ and) (NNS oranges))))")
    >>> e1 = Entity(name="person", start=0, end=5, id="E1")
    >>> e2 = Entity(name="fruit", start=12, end=18, id="E2")
    >>> e3 = Entity(name="fruit", start=23, end=30, id="E3")
    >>> enrich_tree(t, "Alice likes apples and oranges", [e1, e2, e3], [])
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (NP (ENT::fruit apples) (ENT::fruit oranges))))
    """
    assert tree is not None
    assert sentence

    tokens = align_tokens(tree.leaves(), sentence)

    # Insert entities into the tree by length (descending) to handle larger entities first
    computed_spans: set[TREE_POS] = set()
    entities_tree: list[Tree] = []
    for entity in sorted(entities, key=len, reverse=True):
        entity_tokens = tuple(i for i, token in enumerate(tokens) if entity.start <= token[1] and token[0] < entity.end)

        # We do not support two entities at the same place
        if entity_tokens in computed_spans:
            continue

        tree_entity = TreeEntity(entity.name, [tree.leaf_treeposition(i) for i in entity_tokens])
        entity_tree = ins_ent(tree, tree_entity)
        entities_tree.append(entity_tree)
        computed_spans.add(entity_tokens)

    # Unnest any nested entities in reverse order (to avoid modifying parent indices during the process)
    for entity_tree in reversed(entities_tree):
        unnest_ent(entity_tree.parent(), entity_tree.parent_index())

    # Currently, the relation part is commented out, but can be enabled when relations are processed.
    for relation in relations:
        tree_rel = TreeRel((), (), relation.name)
        ins_rel(tree, tree_rel)

    # Remove subtrees that have no specific entity or relation (i.e., generic subtrees)
    for subtree in list(tree.subtrees(lambda x: x.height() == 2 and not has_type(x))):
        subtree.parent().remove(subtree)

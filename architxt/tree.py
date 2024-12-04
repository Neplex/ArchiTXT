import contextlib
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from copy import deepcopy
from enum import Enum
from functools import cache
from typing import Any, TypeAlias, overload

import pandas as pd
from nltk.grammar import Nonterminal, Production
from nltk.tree import ParentedTree

__all__ = [
    'TREE_POS',
    'Forest',
    'NodeLabel',
    'NodeType',
    'Tree',
    'has_type',
    'reduce',
    'reduce_all',
]


TREE_POS = tuple[int, ...]


class NodeType(str, Enum):
    ENT = 'ENT'
    GROUP = 'GROUP'
    REL = 'REL'
    COLL = 'COLL'


class NodeLabel(str):
    type: NodeType
    name: str

    __slots__ = ('name', 'type')

    def __new__(cls, label_type: NodeType, label: str = '') -> 'NodeLabel':
        string_value = f'{label_type.value}::{label}' if label else label_type.value
        return super().__new__(cls, string_value)  # type: ignore

    def __init__(self, label_type: NodeType, label: str = '') -> None:
        self.name = label
        self.type = label_type

    def __reduce__(self) -> tuple[Callable[..., 'NodeLabel'], tuple[Any, ...]]:
        return NodeLabel, (self.type, self.name)


class Tree(ParentedTree):
    slots = ('_parent', '_label')
    _parent: 'Tree | None'
    _label: NodeLabel | str

    def __init__(self, node: NodeLabel | str, children: Iterable['Tree | str'] | None = None) -> None:
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

    def __reduce__(self) -> tuple[Callable[..., 'Tree'], tuple[Any, ...]]:
        return type(self), (self._label, tuple(self))

    @cache
    def height(self) -> int:
        """
        Get the height of the tree.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.height()
        4
        >>> t[0].height()
        3
        >>> t[0, 0].height()
        2

        """
        return super().height()

    @cache
    def depth(self) -> int:
        """
        Get the depth of the tree.

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
        Get the set of group names present within the tree.

        :return: A set of unique group names within the tree.

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
        Get a DataFrame containing all instances of a specified group within the tree.

        Each row in the DataFrame represents an instance of the group, and each column represents an entity in that group, with the value being a concatenated string of that entity's leaves.

        :param group_name: The name of the group to search for.
        :return: A pandas DataFrame containing instances of the specified group.

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
        Get a tuple of subtrees that are entities.

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
        Get the set of entity labels present in the tree.

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
        Return a Counter object that counts the labels of entity subtrees.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.entity_label_count()
        Counter({'person': 2, 'fruit': 1, 'animal': 1})

        """
        return Counter(ent.label().name for ent in self.entities())

    @cache
    def has_duplicate_entity(self) -> bool:
        """
        Check if there are duplicate entity labels.

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
        Check if there is at least one entity as direct children.

        >>> from architxt.tree import Tree
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_entity_child()
        False
        >>> t[0].has_entity_child()
        True

        """
        return any(has_type(child, NodeType.ENT) for child in self)

    def has_unlabelled_nodes(self) -> bool:
        return any(not has_type(subtree) for subtree in self)

    def merge(self, tree: 'Tree') -> 'Tree':
        """
        Merge two trees into one.

        The root of both trees becomes one while maintaining the level of each subtree.
        """
        return type(self)('SENT', deepcopy([*self, *tree]))

    def __reset_cache(self) -> None:
        """Reset cached properties."""
        self.height.cache_clear()
        self.depth.cache_clear()
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
    def __setitem__(self, pos: TREE_POS | int, subtree: 'Tree | str') -> None: ...

    @overload
    def __setitem__(self, pos: slice, subtree: 'list[Tree | str]') -> None: ...

    def __setitem__(self, pos: TREE_POS | int | slice, subtree: 'list[Tree | str] | Tree | str') -> None:
        super().__setitem__(pos, subtree)
        self.__reset_cache()

    def __delitem__(self, pos: TREE_POS | int | slice) -> None:
        super().__delitem__(pos)
        self.__reset_cache()

    def set_label(self, label: NodeLabel | str) -> None:
        super().set_label(label)

        # Do not need to reset our own cache as it does not change our structure
        if parent := self.parent():
            parent.__reset_cache()

    def append(self, child: 'Tree | str') -> None:
        super().append(child)
        self.__reset_cache()

    def extend(self, children: 'Iterable[Tree | str]') -> None:
        super().extend(children)
        self.__reset_cache()

    def remove(self, child: 'Tree | str', *, recursive: bool = True) -> None:
        super().remove(child)
        self.__reset_cache()

        if recursive and len(self) == 0 and (parent := self._parent) is not None:
            parent.remove(self)

    def insert(self, pos: int, child: 'Tree | str') -> None:
        super().insert(pos, child)
        self.__reset_cache()

    def pop(self, pos: int = -1, *, recursive: bool = True) -> 'Tree | str':
        """
        Delete an element from the tree at the specified position `pos`.

        If the parent tree becomes empty after the deletion, recursively deletes the parent node.

        :param pos: The position (index) of the element to delete in the tree.
        :param recursive: If an empty tree should be removed from the parent.
        :return: The element at the position. The function modifies the tree in place.

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
    elif isinstance(t, Nonterminal):
        label = t.symbol()
    else:
        return False

    return isinstance(label, NodeLabel) and label.type.value in types


def reduce(tree: Tree, pos: int, types: set[str | NodeType] | None = None) -> bool:
    """
    Reduces a subtree within a tree at the specified position `pos`.

    The reduction occurs only if the subtree at `pos` has exactly one child,
    or if it does not match a specific set of node types.
    If the subtree can be reduced, its children are lifted into the parent node at `pos`.

    :param tree: The tree in which the reduction will take place.
    :param pos: The index of the subtree to attempt to reduce.
    :param types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: `True` if the subtree was reduced, `False` otherwise.

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
    Recursively attempts to reduce all eligible subtrees in a tree.

    The reduction process continues until no further reductions are possible.
    Subtrees can be skipped if their types are listed in `skip_types`.

    :param tree: The tree in which reductions will be attempted.
    :param skip_types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: None. The tree is modified in place.

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

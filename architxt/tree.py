from __future__ import annotations

import contextlib
import re
import uuid
import weakref
from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    MutableMapping,
    Sequence,
)
from copy import copy, deepcopy
from enum import Enum
from functools import partial, total_ordering
from typing import TYPE_CHECKING, Any, Literal, TextIO, TypeAlias, TypeGuard, overload
from urllib.parse import quote, unquote

import pandas as pd
from cachetools import cachedmethod, keys
from nltk import slice_bounds
from nltk.grammar import Nonterminal, Production
from persistent.list import PersistentList
from persistent.mapping import PersistentMapping
from typing_extensions import Self

from architxt.utils import ExceptionGroup

__all__ = [
    'Forest',
    'NodeLabel',
    'NodeType',
    'Tree',
    'TreeOID',
    'TreePosition',
    'has_type',
    'is_sub_tree',
]

if TYPE_CHECKING:
    __all__ += ['_SubTree', '_TypedSubTree', '_TypedTree']

TreePosition: TypeAlias = tuple[int, ...]
TreeOID: TypeAlias = uuid.UUID

TREE_PARSER_RE = re.compile(r"\(\s*[^\s()]+|[()]|[^\s()]+")


class NodeType(str, Enum):
    ENT = 'ENT'
    GROUP = 'GROUP'
    REL = 'REL'
    COLL = 'COLL'


class NodeLabel(str):
    type: NodeType
    name: str

    __slots__ = ('name', 'type')

    def __new__(cls, label_type: NodeType, label: str = '') -> Self:
        string_value = f'{label_type.value}::{label}' if label else label_type.value
        return super().__new__(cls, string_value)

    def __init__(self, label_type: NodeType, label: str = '') -> None:
        self.name = label
        self.type = label_type

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        return NodeLabel, (self.type, self.name)

    @classmethod
    def fromstring(cls, label: Self | str) -> Self | str:
        if isinstance(label, NodeLabel):
            return label

        if '::' in label:
            node_type, _, name = label.partition('::')
            with contextlib.suppress(ValueError):
                label = NodeLabel(NodeType(node_type), name)

        else:
            with contextlib.suppress(ValueError):
                label = NodeLabel(NodeType(label))

        return label


@total_ordering
class Tree(PersistentList['_SubTree | str']):
    _label: NodeLabel | str
    _metadata: MutableMapping[str, Any]
    _oid: TreeOID

    _v_parent: weakref.ReferenceType[Tree] | None
    _v_cache: MutableMapping[Hashable, Any]

    __slots__ = ('_label', '_metadata', '_oid', '_v_cache', '_v_parent')

    def __init__(
        self,
        label: NodeLabel | str,
        children: Iterable[Tree | str] | None = None,
        metadata: MutableMapping[str, Any] | None = None,
        oid: TreeOID | None = None,
    ) -> None:
        super().__init__(children)
        self._label = NodeLabel.fromstring(label)
        self._metadata = PersistentMapping(metadata or {})
        self._oid = oid or uuid.uuid4()
        self._v_parent = None
        self._v_cache = {}

        self._check_children(self)

        for child in self:
            if isinstance(child, Tree):
                child._v_parent = weakref.ref(self)

    def _check_children(self, children: Iterable[Tree | str]) -> None:
        errors = []

        for index, child in enumerate(children):
            if not isinstance(child, Tree):
                continue

            if self in child.subtrees():
                msg = f'Child at index {index} creates a cyclic reference: a tree cannot contain itself.'
                errors.append(ValueError(msg))

            if is_sub_tree(child) and child.parent is not self:
                msg = f'Child at index {index} is already attached to another parent: {child.parent}.'
                errors.append(ValueError(msg))

        if errors:
            msg = 'Invalid tree children detected'
            raise ExceptionGroup(msg, errors)

    def _invalidate_cache(self) -> None:
        self._v_cache.clear()

        if parent := self.parent:
            parent._invalidate_cache()

    def __eq__(self, other: object) -> bool:
        """
        Compare two subtree objects for equality.

        Two subtrees are considered equal if they have identical labels and identical children (compared recursively).
        The parent reference is not considered in the comparison.

        :param other: The other object to compare against.
        :return: True if the two subtrees are identical in terms of label and children, False otherwise.
        """
        return isinstance(other, Tree) and self.label == other.label and super().__eq__(other)

    def __hash__(self) -> int:
        return self._oid.int

    def __repr__(self) -> str:
        return f'{type(self)}(len={len(self)})'

    def __str__(self) -> str:
        return self.pformat()

    def __setstate__(self, state: object) -> None:
        super().__setstate__(state)
        self._v_parent = None
        self._v_cache = {}

        for child in self:
            if isinstance(child, Tree):
                child._v_parent = weakref.ref(self)

    def __copy__(self) -> Tree:
        """Support for the copy.copy() interface."""
        return self.copy()

    def __deepcopy__(self, _memo: dict[int, Any]) -> Tree:
        """Support for the copy.deepcopy() interface."""
        return self.copy()

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Tree):
            return (len(self), self.label) < (len(other), other.label)

        return str(self.label) < str(other)

    @property
    def oid(self) -> TreeOID:
        return self._oid

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        return self._metadata

    @property
    def parent(self) -> Tree | None:
        """
        The parent of this tree, or None if it has no parent.

        >>> t = Tree.fromstring('(S (A xxx) (A xxx))')
        >>> t.parent

        >>> t[0].parent is t
        True
        >>> t[1].parent is t
        True
        """
        return self._v_parent() if self._v_parent else None

    @property
    def parent_index(self) -> int | None:
        """
        The index of this tree in its parent.

        I.e., ``tree.parent[tree.parent_index] is tree``.
        Note that ``tree.parent_index`` is not necessarily equal to ``tree.parent.index(tree)``,
        since the ``index()`` method returns the first child that is equal to its argument.

        >>> t = Tree.fromstring('(S (A xxx) (A xxx))')
        >>> t.parent_index

        >>> t[0].parent_index
        0
        >>> t[1].parent_index
        1
        """
        if self.parent is None:
            return None

        for i, child in enumerate(self.parent):
            if child is self:
                return i

        msg = "The tree is not in it's parent"
        raise ValueError(msg)

    @property
    def label(self) -> NodeLabel | str:
        """The label of this tree."""
        return self._label

    @label.setter
    def label(self, label: NodeLabel | str) -> None:
        self._label = NodeLabel.fromstring(label)
        self._invalidate_cache()

    @property
    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='root'))
    def root(self) -> Tree:
        """
        The root of this tree.

        I.e., the unique ancestor of this tree whose parent is None.
        If ``tree.parent()`` is None, then ``tree`` is its own root.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t[0, 0].root is t
        True
        """
        node = self
        while (parent := node.parent) is not None:
            node = parent
        return node

    @property
    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='height'))
    def height(self) -> int:
        """
        Get the height of the tree.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.height
        4
        >>> t[0].height
        3
        >>> t[0, 0].height
        2
        """
        return 1 + max((child.height if isinstance(child, Tree) else 1 for child in self), default=0)

    @property
    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='depth'))
    def depth(self) -> int:
        """
        Get the depth of the tree.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.depth
        1
        >>> t[0].depth
        2
        >>> t[0, 0].depth
        3
        """
        return len(self.position) + 1

    @property
    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='position'))
    def position(self) -> TreePosition:
        """
        The tree position of this tree, relative to the root of the tree.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.position
        ()
        >>> t[1, 0].position
        (1, 0)
        """
        if not is_sub_tree(self):
            return ()

        return *self.parent.position, self.parent_index

    def positions(
        self, *, order: Literal['preorder', 'postorder', 'bothorder', 'leaves'] = 'preorder'
    ) -> Generator[TreePosition, None, None]:
        """
        Get all the positions in the tree.

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> list(t.positions()) # doctest: +ELLIPSIS
        [(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
        >>> for pos in t.positions(order='leaves'):
        ...     t[pos] = t[pos][::-1].upper()
        >>> print(t)
        (S (NP (D EHT) (N GOD)) (VP (V DESAHC) (NP (D EHT) (N TAC))))

        :param order: One of: ``preorder``, ``postorder``, ``bothorder``, ``leaves``.
        :yield: All positions in the tree in the given order
        """
        if order in ('preorder', 'bothorder'):
            yield ()

        for i, child in enumerate(self):
            if isinstance(child, Tree):
                yield from ((i, *pos) for pos in child.positions(order=order))
            else:
                yield (i,)

        if order in ('postorder', 'bothorder'):
            yield ()

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='leaves'))
    def leaves(self) -> list[str]:
        """
        Return the leaves of the tree.

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> list(t.leaves())
        ['the', 'dog', 'chased', 'the', 'cat']

        :yield: A list containing this tree's leaves.
            The order reflects the order of the leaves in the tree's hierarchical structure.
        """
        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves())
            else:
                leaves.append(child)

        return leaves

    @overload
    def subtrees(
        self: Self,
        filter_fn: Callable[[Tree], bool] | None = None,
        include_self: Literal[True] = True,
        reverse: bool = False,
    ) -> Iterator[Self]: ...

    @overload
    def subtrees(
        self: Tree,
        filter_fn: Callable[[Tree], bool] | None = None,
        include_self: Literal[False] = False,
        reverse: bool = False,
    ) -> Iterator[_SubTree]: ...

    def subtrees(
        self,
        filter_fn: Callable[[Tree], bool] | None = None,
        include_self: bool = True,
        reverse: bool = False,
    ) -> Iterator[Tree]:
        """
        Get all the subtrees of this tree, optionally restricted to trees matching the filter function.

        :param filter_fn: The function to filter all local trees
        :param include_self: Whether to include this tree in the output.
        :param reverse: Whether to traverse tree from right to left and bottom-up.

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> for s in t.subtrees(lambda t: t.height == 2):
        ...     print(s)
        (D the)
        (N dog)
        (V chased)
        (D the)
        (N cat)
        >>> for s in t.subtrees(lambda t: t.height == 2, reverse=True):
        ...     print(s)
        (N cat)
        (D the)
        (V chased)
        (N dog)
        (D the)
        """
        if reverse:
            for child in reversed(self):
                if isinstance(child, Tree):
                    yield from child.subtrees(filter_fn, include_self=True, reverse=reverse)

        if include_self and (not filter_fn or filter_fn(self)):
            yield self

        if not reverse:
            for child in self:
                if isinstance(child, Tree):
                    yield from child.subtrees(filter_fn, include_self=True, reverse=reverse)

    def productions(self) -> list[Production]:
        """
        Generate the productions that correspond to the non-terminal nodes of the tree.

        For each subtree of the form (P: C1 C2 ... Cn) this produces a production of the form P -> C1 C2 ... Cn.

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.productions() # doctest: +NORMALIZE_WHITESPACE
        [S -> NP VP, NP -> D N, D -> 'the', N -> 'dog', VP -> V NP, V -> 'chased',
        NP -> D N, D -> 'the', N -> 'cat']
        """
        child_names = [Nonterminal(child.label) if isinstance(child, Tree) else child for child in self]
        productions = [Production(Nonterminal(self.label), child_names)]

        for child in self:
            if isinstance(child, Tree):
                productions.extend(child.productions())

        return productions

    def leaf_position(self, index: int) -> TreePosition:
        """
        Return the tree position of the `index`-th leaf in this tree.

        The tree position is a tuple of indices that corresponds to the
        location of the `index`-th leaf in the tree structure.
        If `tp = self.leaf_position(i)`, then `self[tp]` should be
        the same as `self.leaves()[i]`.

        :param index: The index of the leaf for which to find the tree position.
        :returns: A tuple representing the tree position of the `index`-th leaf.
        :raise IndexError: If `index` is negative or if there are fewer than `index + 1` leaves in the tree.

        >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        >>> t.leaf_position(0)
        (0, 0, 0)
        >>> t.leaf_position(4)
        (1, 1, 1, 0)
        """
        if index < 0:
            msg = 'index must be non-negative'
            raise IndexError(msg)

        stack: list[tuple[Tree | str, tuple[int, ...]]] = [(self, ())]
        count = 0

        while stack:
            node, pos = stack.pop()
            if isinstance(node, Tree):
                # Add children in reverse to simulate left-to-right traversal
                stack.extend((node[i], (*pos, i)) for i in reversed(range(len(node))))
            else:
                if count == index:
                    return pos
                count += 1

        msg = "index must be less than or equal to len(self)"
        raise IndexError(msg)

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='groups'))
    def groups(self) -> set[str]:
        """
        Get the set of group names present within the tree.

        :return: A set of unique group names within the tree.

        >>> t = Tree.fromstring('(S (GROUP::A x) (GROUP::B y) (X (GROUP::C z)))')
        >>> sorted(t.groups())
        ['A', 'B', 'C']
        >>> sorted(t[0].groups())
        ['A']

        """
        result = set()

        if isinstance(self.label, NodeLabel) and has_type(self, NodeType.GROUP):
            result.add(self.label.name)

        for child in self:
            if isinstance(child, Tree):
                result.update(child.groups())

        return result

    def group_instances(self, group_name: str) -> pd.DataFrame:
        """
        Get a DataFrame containing all instances of a specified group within the tree.

        Each row in the DataFrame represents an instance of the group, and each column represents an entity in that
        group, with the value being a concatenated string of that entity's leaves.

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
        >>> t.group_instances("C")
        Empty DataFrame
        Columns: []
        Index: []
        >>> t[0].group_instances("A")
          person  fruit
        0  Alice  apple
        """
        dataframes = [child.group_instances(group_name) for child in self if isinstance(child, Tree)]

        if isinstance(self.label, NodeLabel) and has_type(self, NodeType.GROUP) and self.label.name == group_name:
            root_dataframe = pd.DataFrame(
                [
                    {
                        sub_child.label.name: ' '.join(sub_child.leaves())
                        for sub_child in self
                        if has_type(sub_child, NodeType.ENT)
                    }
                ]
            )
            dataframes.append(root_dataframe)

        if not dataframes:
            return pd.DataFrame()

        return pd.concat(dataframes, ignore_index=True).drop_duplicates()

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='entities'))
    def entities(self) -> tuple[_TypedTree, ...]:
        """
        Get a tuple of subtrees that are entities.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> list(t.entities()) == [t[0, 0], t[0, 1], t[1, 0], t[1, 1]]
        True
        >>> del t[0]
        >>> list(t.entities()) == [t[0, 0], t[0, 1]]
        True
        >>> list(t[0, 0].entities()) == [t[0, 0]]
        True
        """
        return tuple(ent for ent in self.subtrees() if has_type(ent, NodeType.ENT))

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='entity_labels'))
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
        return {ent.label.name for ent in self.entities()}

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='entity_label_count'))
    def entity_label_count(self) -> Counter[str]:
        """
        Return a Counter object that counts the labels of entity subtrees.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.entity_label_count()
        Counter({'person': 2, 'fruit': 1, 'animal': 1})
        """
        return Counter(ent.label.name for ent in self.entities())

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='has_duplicate_entity'))
    def has_duplicate_entity(self) -> bool:
        """
        Check if there are duplicate entity labels.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_duplicate_entity()
        True
        >>> t[0].has_duplicate_entity()
        False
        """
        return any(v > 1 for v in self.entity_label_count().values())

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='has_entity_child'))
    def has_entity_child(self) -> bool:
        """
        Check if there is at least one entity as direct children.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_entity_child()
        False
        >>> t[0].has_entity_child()
        True
        """
        return any(has_type(child, NodeType.ENT) for child in self)

    @cachedmethod(lambda self: self._v_cache, key=partial(keys.methodkey, method='has_unlabelled_nodes'))
    def has_unlabelled_nodes(self) -> bool:
        """
        Check if any child has a non-typed label.

        :return: A boolean indicating if the node contains any non-typed label.

        >>> t = Tree.fromstring('(S (X xxx) (Y yyy) (Z zzz))')
        >>> t.has_unlabelled_nodes()
        True

        >>> t = Tree.fromstring('(S (ENT::X xxx) (REL::Y yyy) (COLL::Z zzz))')
        >>> t.has_unlabelled_nodes()
        False
        """
        return any(not has_type(subtree) for subtree in self)

    def merge(self, tree: Tree) -> Tree:
        """
        Merge two trees into one.

        The root of both trees becomes one while maintaining the level of each subtree.
        """
        children: list[Tree] = []

        if self.label == 'ROOT':
            children.extend(self)
        else:
            children.append(self)

        if tree.label == 'ROOT':
            children.extend(tree)
        else:
            children.append(tree)

        return type(self)('ROOT', deepcopy(children))

    def reduce(self, skip_types: set[str | NodeType] | None = None) -> bool:
        """
        Attempt to reduce this subtree, lifting the children into the parent node, replacing the subtree.

        Reduction happens if:
        - The tree has exactly one child, AND
        - Its label is not in `types` (if `types` is given)

        :param skip_types: A set of node types that should be kept, or `None` to reduce reduces all single-child nodes.
        :return: `True` if the subtree was reduced, `False` otherwise.

        >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
        >>> t[1, 1].reduce()
        True
        >>> print(t)
        (S (NP Alice) (VP (VB like) (NNS apples)))
        >>> t[0].reduce()
        True
        >>> print(t)
        (S Alice (VP (VB like) (NNS apples)))
        """
        if not is_sub_tree(self) or len(self) != 1 or (skip_types and has_type(self, skip_types)):
            return False

        # Replace the original subtree by its children into the parent at `pos`
        parent_index = self.parent_index
        self.parent[parent_index : parent_index + 1] = [
            child.detach() if isinstance(child, Tree) else child for child in self
        ]

        return True

    def reduce_all(self, skip_types: set[str | NodeType] | None = None) -> None:
        """
        Recursively reduces all reducible subtrees in the tree.

        The reduction process continues until no further reductions are possible.
        Subtrees can be skipped if their types are listed in `skip_types`.

        :param skip_types: A set of node types that should be kept, or `None` to reduce reduces all single-child nodes.

        >>> t = Tree.fromstring("(S (X (Y (Z (NP Alice)))) (VP (VB likes) (NP (NNS apples))))")
        >>> t.reduce_all()
        >>> print(t)
        (S Alice (VP likes apples))
        """
        reduced = True
        while reduced:
            reduced = False

            for subtree in self.subtrees():
                if subtree.reduce(skip_types=skip_types):
                    reduced = True
                    break

    @overload
    def __getitem__(self, pos: tuple[()]) -> Self: ...

    @overload
    def __getitem__(self, pos: TreePosition | int) -> _SubTree | str: ...

    @overload
    def __getitem__(self, pos: slice) -> list[_SubTree | str]: ...

    def __getitem__(self, pos: TreePosition | int | slice) -> Tree | str | list[_SubTree | str]:
        """
        Retrieve a child or subtree using an index, a slice, or a tree position.

        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> print(t[()])
        (S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))
        >>> print(t[0])
        (X (ENT::person Alice) (ENT::fruit apple))
        >>> print(t[0, 1])
        (ENT::fruit apple)
        >>> print(t[1:][0])
        (Y (ENT::person Bob) (ENT::animal rabbit))
        """
        if isinstance(pos, int | slice):
            # We access `data` directly instead of using `super()` because `UserList` casts slice outputs
            # to the parent class, which would return a Tree instead of a plain list.
            return self.data[pos]

        if not isinstance(pos, tuple):
            msg = f'indices must be integers, slices or tuple of int, not {type(pos).__name__}'
            raise TypeError(msg)

        node = self
        for depth, idx in enumerate(pos):
            if not isinstance(idx, int):
                msg = f'multi-level indices must be integers, not {type(pos).__name__}'
                raise TypeError(msg)

            if not isinstance(node, Tree):
                msg = f'index {idx} out of range at position {pos[:depth]} (leaf node reached)'
                raise IndexError(msg)

            try:
                node = node.data[idx]
            except IndexError:
                msg = f'index {idx} out of range at position {pos[:depth]}'
                raise IndexError(msg)

        return node

    @overload
    def __setitem__(self, pos: TreePosition | int, subtree: Tree | str) -> None: ...

    @overload
    def __setitem__(self, pos: slice, subtree: Iterable[Tree | str]) -> None: ...

    def __setitem__(self, pos: TreePosition | int | slice, subtree: Tree | str | Iterable[Tree | str]) -> None:  # noqa: C901
        # ptree[start:stop] = subtree
        if isinstance(pos, slice):
            start, stop, step = slice_bounds(self, pos, allow_step=True)
            # make a copy of subtree, in case it's an iterator
            if not isinstance(subtree, list | tuple):
                subtree = list(subtree)
            # Check for any error conditions, so we can avoid ending
            # up in an inconsistent state if an error does occur.
            self._check_children(subtree)
            # clear the child pointers of all parents we're removing
            for i in range(start, stop, step):
                if isinstance((child := self[i]), Tree):
                    child._v_parent = None
            # set the child pointers of the new children. We do this
            # after clearing *all* child pointers, in case we're e.g.
            # reversing the elements in a tree.
            for i, child in enumerate(subtree):
                if isinstance(child, Tree):
                    child._v_parent = weakref.ref(self)
            # finally, update the content of the child list itself.
            super().__setitem__(pos, subtree)

        # ptree[i] = subtree
        elif isinstance(pos, int):
            if pos < 0:
                pos += len(self)
            if pos < 0:
                msg = 'pos out of range'
                raise IndexError(msg)
            # if the subtree is not changing, do nothing.
            if subtree is self[pos]:
                return
            # Set the new child's parent pointer.
            if isinstance(subtree, Tree):
                subtree._v_parent = weakref.ref(self)
            # Remove the old child's parent pointer
            if isinstance((child := self[pos]), Tree):
                child._v_parent = None
            # Update our child list.
            super().__setitem__(pos, subtree)

        elif isinstance(pos, tuple):
            if not isinstance(subtree, Tree | str):
                msg = f'subtree must be a Tree or str, not {type(subtree).__name__}'
                raise TypeError(msg)

            # ptree[()] = subtree
            if len(pos) == 0:
                msg = 'position () may not be assigned to'
                raise IndexError(msg)

            # ptree[i1, i2, i3] = subtree
            node = self[pos[:-1]]
            if not isinstance(node, Tree):
                msg = f'index {pos[-1]} out of range at position {pos[:-1]} (leaf node reached)'
                raise IndexError(msg)
            node[pos[-1]] = subtree

        else:
            msg = f'indices must be integers, slices or tuple of int, not {type(pos).__name__}'
            raise TypeError(msg)

        self._invalidate_cache()

    def __delitem__(self, pos: TreePosition | int | slice) -> None:  # noqa: C901
        # del ptree[start:stop]
        if isinstance(pos, slice):
            start, stop, step = slice_bounds(self, pos, allow_step=True)
            # Clear all the children pointers.
            for i in range(start, stop, step):
                if isinstance((child := self[i]), Tree):
                    child._v_parent = None
            # Delete the children from our child list.
            super().__delitem__(pos)

        # del ptree[i]
        elif isinstance(pos, int):
            if pos < 0:
                pos += len(self)
            if pos < 0:
                msg = 'pos out of range'
                raise IndexError(msg)
            # Clear the child's parent pointer.
            if isinstance((child := self[pos]), Tree):
                child._v_parent = None
            # Remove the child from our child list.
            super().__delitem__(pos)

        elif isinstance(pos, tuple):
            # del ptree[()]
            if len(pos) == 0:
                msg = 'position () may not be deleted'
                raise IndexError(msg)

            # del ptree[i1, i2, i3]
            node = self[pos[:-1]]
            if not isinstance(node, Tree):
                msg = f'index {pos[-1]} out of range at position {pos[:-1]} (leaf node reached)'
                raise IndexError(msg)
            del node[pos[-1]]

        else:
            msg = f'indices must be integers, slices or tuple of int, not {type(pos).__name__}'
            raise TypeError(msg)

        self._invalidate_cache()

    def clear(self) -> None:
        super().clear()
        self._invalidate_cache()

    def append(self, child: Tree | str) -> None:
        if isinstance(child, Tree):
            self._check_children([child])
            child._v_parent = weakref.ref(self)

        super().append(child)
        self._invalidate_cache()

    def extend(self, children: Iterable[Tree | str]) -> None:
        # Convert to list only if it's a one-shot iterable (like a generator)
        if not isinstance(children, Collection):
            children = list(children)

        self._check_children(children)
        for child in children:
            if isinstance(child, Tree):
                child._v_parent = weakref.ref(self)

        super().extend(children)
        self._invalidate_cache()

    def remove(self, child: _SubTree | str, *, recursive: bool = True) -> None:
        super().remove(child)

        if isinstance(child, Tree):
            child._v_parent = None

        if recursive and len(self) == 0 and is_sub_tree(self):
            self.parent.remove(self)

        self._invalidate_cache()

    def insert(self, pos: int, child: Tree | str) -> None:
        # Set the child's parent and update our child list.
        if isinstance(child, Tree):
            self._check_children([child])
            child._v_parent = weakref.ref(self)

        super().insert(pos, child)
        self._invalidate_cache()

    def pop(self, pos: int = -1, *, recursive: bool = True) -> Tree | str:
        """
        Delete an element from the tree at the specified position `pos`.

        If the parent tree becomes empty after the deletion, parent nodes are recursively deleted.

        :param pos: The position (index) of the element to delete in the tree.
        :param recursive: If an empty tree should be removed from the parent.
        :return: The element at the position. The function modifies the tree in place.

        >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
        >>> print(t[(1, 1)])
        (NP (NNS apples))
        >>> subtree = t[1, 1].pop(0)
        >>> print(t)
        (S (NP Alice) (VP (VB like)))
        >>> subtree = t.pop(0)
        >>> print(t)
        (S (VP (VB like)))
        >>> subtree = t[0].pop(0, recursive=False)
        >>> print(t)
        (S (VP ))

        """
        child = super().pop(pos)

        if isinstance(child, Tree):
            child._v_parent = None

        if recursive and len(self) == 0 and is_sub_tree(self):
            self.parent.remove(self)

        self._invalidate_cache()

        return child

    def detach(self) -> Tree:
        """
        Detach a subtree from its parent.

        :return: The detached tree.

        >>> t = Tree.fromstring('(S (A xxx) (B yyy))')
        >>> detached = t[0].detach()
        >>> print(detached.root)
        (A xxx)
        >>> print(t)
        (S (B yyy))
        """
        if is_sub_tree(self):
            self.parent.remove(self, recursive=False)

        return self

    def copy(self) -> Tree:
        """
        Copy an entire tree.

        :return: A new copy of the tree.
        """
        return Tree(
            label=copy(self.label),
            children=[child.copy() if isinstance(child, Tree) else child for child in self],
            metadata=self.metadata,
            oid=self.oid,
        )

    @classmethod
    def fromstring(cls, text: str) -> Tree:
        """
        Read a tree from a LISP-style notation.

        Trees are represented as nested brackettings, such as:

          (S (NP (NNP John)) (VP (V runs)))

        :param text: The string to read
        :return: A tree corresponding to the string representation ``text``.

        >>> t = Tree.fromstring('(S (X xxx) (Y yyy))')
        >>> print(t)
        (S (X xxx) (Y yyy))
        """
        # Walk through each token, updating a stack of trees.
        stack: list[tuple[str | None, list[Any]]] = [(None, [])]  # list of (node, children) tuples
        label: str | None

        for match in TREE_PARSER_RE.finditer(text):
            token = match.group()

            # Beginning of a tree/subtree
            if token.startswith('('):
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(text, 'end-of-string', match)
                label = unquote(token[1:].lstrip())
                stack.append((label, []))

            # End of a tree/subtree
            elif token == ')':
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(text, '(', match)
                    else:
                        cls._parse_error(text, 'end-of-string', match)
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))

            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(text, '(', match)
                stack[-1][1].append(unquote(token))

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(text, ')')
        elif len(stack[0][1]) == 0:
            cls._parse_error(text, '(')
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1

        return stack[0][1][0]

    @classmethod
    def _parse_error(cls, text: str, expecting: str, match: re.Match[str] | None = None) -> None:
        """
        Display a friendly error message when parsing a tree string fails.

        :param text: The string we're parsing.
        :param expecting: What we expected to see instead.
        :param match: Regexp match of the problem token or `None` if end-of-string.
        """
        # Construct a basic error message
        if match:
            pos, token = match.start(), match.group()
        else:
            pos, token = len(text), 'end-of-string'

        msg = f"{cls.__name__}.read(): expected {expecting!r} but got {token!r}\n{' ' * 12}at index {pos}."

        # Add a display showing the error token itself:
        text = text.replace("\n", " ").replace("\t", " ")
        offset = pos
        if len(text) > pos + 10:
            text = text[: pos + 10] + "..."
        if pos > 10:
            text = "..." + text[pos - 10 :]
            offset = 13

        msg += '\n{}"{}"\n{}^'.format(" " * 16, text, " " * (17 + offset))
        raise ValueError(msg)

    def pretty_print(
        self, highlight: Sequence[Tree | int] = (), stream: TextIO | None = None, maxwidth: int = 32
    ) -> None:
        """
        Pretty-print this tree as ASCII or Unicode art.

        It relies on :py:class:`nltk.tree.prettyprinter.TreePrettyPrinter`.

        :param stream:  The file to print to.
        :param highlight: Optionally, a sequence of Tree objects in `tree` which should be highlighted.
            Has the effect of only applying colors to nodes in this sequence.
        :param maxwidth: maximum number of characters before a label starts to wrap.
            Leaf with more than `maxwidth` characters will be truncated.
        """
        from nltk.tree import Tree as NLTKTree
        from nltk.tree.prettyprinter import TreePrettyPrinter

        nltk_tree = NLTKTree.fromstring(str(self))

        for tree_position in nltk_tree.treepositions(order='leaves'):
            if len(nltk_tree[tree_position]) > maxwidth:
                nltk_tree[tree_position] = nltk_tree[tree_position][: maxwidth - 3] + '...'

        print(TreePrettyPrinter(nltk_tree, highlight=highlight).text(unicodelines=True, maxwidth=maxwidth), file=stream)

    def to_svg(self, highlight: Sequence[Tree | int] = ()) -> str:
        """
        Pretty-print this tree as SVG.

        It relies on :py:class:`nltk.tree.prettyprinter.TreePrettyPrinter`.

        :param highlight: Optionally, a sequence of Tree objects in `tree` which should be highlighted.
            Has the effect of only applying colors to nodes in this sequence.
        """
        from nltk.tree import Tree as NLTKTree
        from nltk.tree.prettyprinter import TreePrettyPrinter

        nltk_tree = NLTKTree.fromstring(str(self))
        return TreePrettyPrinter(nltk_tree, highlight=highlight).svg()

    def pformat(self, margin: int | None = None, indent: int = 0) -> str:
        """
        Get a pretty-printed string representation of this tree.

        :param margin: The right margin at which to do line-wrapping.
        :param indent: The indentation level at which printing begins.
        :return: A pretty-printed string representation of this tree.

        >>> t = Tree('S', [Tree('X', ['xxx']), Tree('Y', ['yyy'])])
        >>> t.pformat()
        '(S (X xxx) (Y yyy))'
        """
        pad = ' ' * indent
        text = f"{pad}({self._label} {' '.join(str(child) if isinstance(child, Tree) else quote(child) for child in self)})"

        if margin is None or len(text) + indent < margin:
            return text

        child_lines = '\n'.join(
            child.pformat(margin, indent + 2) if isinstance(child, Tree) else f'{pad}  {quote(child)}' for child in self
        )
        return f"{pad}({self._label} {child_lines})"

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the tree into a JSON-serializable dictionary.

        :return: A dictionary containing the JSON-serializable representation with keys:
            - `oid`: string representation of the tree OID.
            - `type`: node type name (e.g., "ENT", "GROUP") when the node is typed, or None.
            - `name`: node name when typed, otherwise the raw label value.
            - `metadata`: a plain dict containing the node's metadata.
            - `children`: list of child entries where each child is either a serialized child
                dictionary (for subtree children) or the leaf value.
        """
        return {
            'oid': str(self.oid),
            'type': self.label.type.value if has_type(self) else None,
            'name': self.label.name if has_type(self) else self.label,
            'metadata': dict(self.metadata),
            'children': [child.to_json() if isinstance(child, Tree) else child for child in self],
        }

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Tree:
        """
        Construct a Tree from a JSON-like mapping containing keys 'oid', 'type', 'name', 'metadata', and 'children'.

        Expected keys:
        - `oid` (optional): UUID string; parsed into a :py:class:`~uuid.UUID` when provided. Default to random one.
        - `type` (optional): string name of a NodeType; if present a typed NodeLabel is created.
        - `name` (optional): node name or label (defaults to empty string when absent).
        - `metadata` (optional): mapping stored as the Tree's metadata.
        - `children` (optional): list of either leaf strings or nested child mappings to be recursively converted.

        :param json_data: Dictionary containing the JSON-serializable representation of the tree.
        :return: The reconstructed Tree instance.
        :raises ValueError: If the JSON-like mapping is invalid.
        """
        try:
            label: NodeLabel | str = json_data.get('name') or ''

            if tree_type := json_data.get('type'):
                label = NodeLabel(NodeType(tree_type), label)

            oid = json_data.get('oid')
            children = [
                child if isinstance(child, str) else cls.from_json(child) for child in json_data.get('children', [])
            ]

            return cls(
                label=label,
                oid=uuid.UUID(oid) if oid else None,
                metadata=json_data.get('metadata'),
                children=children,
            )

        except Exception as e:
            msg = f'Failed to parse JSON data: {json_data}'
            raise ValueError(msg) from e


if TYPE_CHECKING:

    class _SubTree(Tree):
        parent: Tree
        parent_index: int

    class _TypedTree(Tree):
        @property
        def label(self) -> NodeLabel: ...

        @label.setter
        def label(self, value: NodeLabel | str) -> None: ...

    class _TypedSubTree(_SubTree, _TypedTree): ...


@overload
def is_sub_tree(t: _TypedTree) -> TypeGuard[_TypedSubTree]: ...


@overload
def is_sub_tree(t: Tree) -> TypeGuard[_SubTree]: ...


def is_sub_tree(tree: Tree) -> bool:
    """
    Determine whether the given Tree instance is a subtree.

    This helper function serves as a type guard to assist static type checkers
    like mypy in refining the type of `tree` when the function returns True.

    :param tree: The tree instance to check.
    :return: True if `tree` is a subtree (i.e., has a parent), False otherwise.

    >>> t = Tree.fromstring('(S (X xxx) (Y yyy))')
    >>> is_sub_tree(t)
    False
    >>> is_sub_tree(t[0])
    True

    """
    return tree.parent is not None


@overload
def has_type(t: _SubTree, types: set[NodeType | str] | NodeType | str | None = None) -> TypeGuard[_TypedSubTree]: ...


@overload
def has_type(t: Any, types: set[NodeType | str] | NodeType | str | None = None) -> TypeGuard[_TypedTree]: ...


def has_type(t: Any, types: set[NodeType | str] | NodeType | str | None = None) -> bool:
    """
    Check if the given tree object has the specified type(s).

    :param t: The object to check type for (can be a Tree, Production, or NodeLabel).
    :param types: The types to check for (can be a set of strings, a string, or None).
    :return: True if the object has the specified type(s), False otherwise.

    >>> tree = Tree.fromstring('(S (ENT Alice) (REL Bob))')
    >>> has_type(tree, NodeType.ENT)
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
    # Normalize type input
    if types is None:
        types = set(NodeType)
    elif not isinstance(types, set):
        types = {types}

    types = {t.value if isinstance(t, NodeType) else str(t) for t in types}
    label: NodeLabel | str

    # Check for the type in the respective object
    if isinstance(t, NodeLabel):
        label = t
    elif isinstance(t, Tree):
        label = t.label
    elif isinstance(t, Production):
        label = t.lhs().symbol()
    elif isinstance(t, Nonterminal):
        label = t.symbol()
    else:
        return False

    return isinstance(label, NodeLabel) and label.type.value in types


Forest: TypeAlias = Collection[Tree]

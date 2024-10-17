import contextlib
from collections import Counter
from collections.abc import Collection, Generator, Iterable
from functools import cached_property
from typing import TypeAlias, TypeVar

from nltk.grammar import Production
from nltk.tokenize.util import align_tokens
from nltk.tree import ParentedTree as NLTKParentedTree
from nltk.tree import Tree as NLTKTree

from architxt.model import TREE_POS, Entity, NodeLabel, NodeType, Relation, TreeEntity, TreeRel

__all__ = [
    'has_type',
    'Tree',
    'ParentedTree',
    'Forest',
    'ins_elem',
    'del_elem',
    'reduce',
    'reduce_all',
    'fix_coord',
    'fix_conj',
    'fix_all_coord',
    'ins_ent',
    'ins_rel',
    'enrich_tree',
    'unnest_ent',
    'update_cache',
]


class Tree(NLTKTree):
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

    def __reduce__(self):
        return self.__class__, (str(self.label()), tuple(self))

    @cached_property
    def entities(self) -> Generator['Tree', None, None]:
        """
        Returns a generator for subtrees that are entities.

        Example:
        >>> t = ParentedTree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> list(t.entities) == [t[0, 0], t[0, 1], t[1, 0], t[1, 1]]
        True
        """
        for child in self:
            if has_type(child, NodeType.ENT):
                yield child

            if isinstance(child, Tree):
                yield from child.entities

    @cached_property
    def entity_labels(self) -> set[str]:
        """
        Return a set of entity labels present in the tree.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> sorted(t.entity_labels)
        ['animal', 'fruit', 'person']
        """
        return {node.label().name for node in self.entities}

    @cached_property
    def entity_label_count(self) -> Counter[NodeLabel]:
        """
        Returns a Counter object that counts the labels of entity subtrees.

        Example:
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.entity_label_count
        Counter({'person': 2, 'fruit': 1, 'animal': 1})
        """
        return Counter(ent.label().name for ent in self.entities)

    @cached_property
    def has_duplicate_entity(self) -> bool:
        """
        Checks if there are duplicate entity labels.

        Example:
        >>> from architxt.tree import Tree
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_duplicate_entity
        True
        >>> t[0].has_duplicate_entity
        False
        """
        return any(v > 1 for v in self.entity_label_count.values())

    @cached_property
    def has_entity_child(self) -> bool:
        """
        Checks if there is at least one entity as direct children.

        Example:
        >>> from architxt.tree import Tree
        >>> t = Tree.fromstring('(S (X (ENT::person Alice) (ENT::fruit apple)) (Y (ENT::person Bob) (ENT::animal rabbit)))')
        >>> t.has_entity_child
        False
        >>> t[0].has_entity_child
        True
        """
        return any(has_type(child, NodeType.ENT) for child in self)

    def merge(self, tree: NLTKTree) -> 'Tree':
        """
        Merge two trees into one.
        The root of both tree become one while maintaining the level of each subtree.
        """
        return self.__class__(
            'ROOT',
            [
                *self.__class__.convert(self),
                *self.__class__.convert(tree),
            ],
        )

    def __repr__(self) -> str:
        return f'{self.__class__}(len={len(self)})'


class ParentedTree(Tree, NLTKParentedTree):
    def depth(self) -> int:
        return len(self.treeposition()) + 1


_Tree = TypeVar('_Tree', bound=Tree)
Forest: TypeAlias = Collection[_Tree]


def update_cache(x: ParentedTree) -> None:
    """
    This method updates the similarity cache.
    It removes any cache entries that contain the tree.
    """
    from architxt.similarity import SIM_CACHE, SIM_CACHE_LOCK

    position = x.treeposition()

    with SIM_CACHE_LOCK:
        keys_to_remove = {key for key in SIM_CACHE if key[0] == position or key[1] == position}

        for key in keys_to_remove:
            del SIM_CACHE[key]


def has_type(t: Tree | Production | NodeLabel, types: set[NodeType | str] | NodeType | str | None = None) -> bool:
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

    # Normalize types input
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


def ins_elem(t: ParentedTree, x: Tree, pos: int) -> None:
    """
    Inserts a tree element `x` into the parented tree `t` at the specified position `pos`.

    :param t: The `ParentedTree` in which the element is to be inserted.
    :param x: The `Tree` element to insert, which will be converted to a `ParentedTree`.
    :param pos: The position (index) where the new element `x` will be inserted in the tree.
    :return: None. The function modifies the input tree `t` in place.

    Example:
    >>> t = ParentedTree.fromstring("(S (VP (VB like) (NP (NNS apples))))")
    >>> x = Tree("NP", ["Alice"])
    >>> ins_elem(t, x, 0)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB like) (NP (NNS apples))))
    """
    assert t is not None

    # Insert the element `x` into the tree `t` at the specified position `pos`
    t.insert(pos, ParentedTree.convert(x))

    # Update the cache for the inserted subtree
    update_cache(t)


def del_elem(t: ParentedTree, pos: int) -> None:
    """
    Deletes an element from the parented tree `t` at the specified position `pos`.
    If the parent tree becomes empty after the deletion, recursively deletes the parent node.

    :param t: The `ParentedTree` from which an element is to be deleted.
    :param pos: The position (index) of the element to delete in the tree.
    :return: None. The function modifies the input tree `t` in place.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> print(t[(1, 1)].pformat(margin=255))
    (NP (NNS apples))
    >>> del_elem(t[(1, 1)], 0)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB like)))
    >>> del_elem(t, 0)
    >>> print(t.pformat(margin=255))
    (S (VP (VB like)))
    >>> del_elem(t, 0)
    >>> print(t.pformat(margin=255))
    (S )
    """
    assert t is not None

    # Remove the element at the given position `pos`
    t.pop(pos)

    # Update the cache for the current tree
    update_cache(t)

    # If the tree becomes empty after the deletion, recursively delete the parent node
    if len(t) == 0 and t.parent() is not None:
        del_elem(t.parent(), t.parent_index())


def reduce(t: ParentedTree, pos: int, types: set[str | NodeType] | None = None) -> bool:
    """
    Reduces a subtree within a `ParentedTree` at the specified position `pos`. The reduction occurs only
    if the subtree at `pos` has exactly one child, or if it don't match a specific set of node types.
    If the subtree  can be reduced, its children are lifted into the parent node at `pos`.

    :param t: The `ParentedTree` in which the reduction will take place.
    :param pos: The index of the subtree to attempt to reduce.
    :param types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: `True` if the subtree was reduced, `False` otherwise.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> reduce(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB like) (NNS apples)))
    >>> reduce(t, 0)
    True
    >>> print(t.pformat(margin=255))
    (S Alice (VP (VB like) (NNS apples)))
    """
    assert t is not None

    # Check if the tree at the specified position can be reduced
    if (
        not isinstance(t[pos], Tree)  # Ensure the subtree at `pos` is a Tree
        or (types and has_type(t[pos], types))  # Check if it matches the specified types
        or (not types and len(t[pos]) > 1)  # If no types, only reduce if it has one child
    ):
        return False

    # Convert children of the subtree into a list of `Tree` objects
    children = [Tree.convert(child) for child in t[pos]]

    # Remove the subtree at position `pos` and update the cache
    t.pop(pos)
    update_cache(t)

    # Insert each child of the original subtree into the parent at `pos`
    for i, child in enumerate(children):
        t.insert(pos + i, ParentedTree.convert(child))

    return True


def reduce_all(t: ParentedTree, skip_types: set[str | NodeType] | None = None) -> None:
    """
    Recursively attempts to reduce all eligible subtrees in a `ParentedTree`. The reduction process continues
    until no further reductions are possible. Subtrees can be skipped if their types are listed in `skip_types`.

    :param t: The `ParentedTree` in which reductions will be attempted.
    :param skip_types: A set of `NodeType` or string labels that should be kept, or `None` to reduce based on length.
    :return: None. The tree `t` is modified in place.

    Example:
    >>> t = ParentedTree.fromstring("(S (X (Y (Z (NP Alice)))) (VP (VB likes) (NP (NNS apples))))")
    >>> reduce_all(t)
    >>> print(t.pformat(margin=255))
    (S Alice (VP likes apples))
    """
    assert t is not None

    reduced = True
    while reduced:
        reduced = False

        for subtree in t.subtrees(lambda st: isinstance(st, ParentedTree) and st.parent() is not None):
            if reduce(subtree.parent(), subtree.parent_index(), types=skip_types):
                reduced = True
                break


def fix_coord(t: ParentedTree, pos: int) -> bool:
    """
    Fixes the coordination structure in a `ParentedTree` at the specified position `pos`.
    This function modifies the tree to ensure that the conjunctions are structured correctly
    according to the grammar rules of coordination.

    :param t: The `ParentedTree` in which coordination adjustments will be made.
    :param pos: The index of the subtree within the parent tree that contains the coordination to fix.
    :return: `True` if the coordination was successfully fixed, `False` otherwise.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_coord(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))
    """
    assert t is not None

    coord = None

    # Identify the coordination subtree
    for child in t[pos]:
        if (
            isinstance(child, ParentedTree)
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
    left = Tree(t[pos].label(), children=[Tree.convert(child) for child in t[pos][:coord_index]])
    right = [Tree.convert(child) for child in coord[1:]]  # Get all NPs after the conjunction

    # Create the conjunction tree
    conj = Tree('CONJ', children=[left, *right])  # CONJ should include the left NP and the conjuncts

    # Insert the new structure back into the tree
    # If children remains on the right of the coordination, we keep the existing level
    new_tree = (
        Tree(t[pos].label(), children=[conj] + [Tree.convert(child) for child in remaining_children])
        if (remaining_children := t[pos][coord_index + 1 :])
        else conj
    )

    # Replace the old subtree
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree))

    return True


def fix_conj(t: ParentedTree, pos: int) -> bool:
    """
    Fixes conjunction structures in a `ParentedTree` at the specified position `pos`.
    If the node at `pos` is labeled 'CONJ', the function flattens any nested conjunctions
    by replacing the node with a new tree that combines its children.

    :param t: The `ParentedTree` in which the conjunction structure will be fixed.
    :param pos: The index of the 'CONJ' node to be processed.

    :return: `True` if the conjunction structure was modified, `False` otherwise.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))")
    >>> fix_conj(t[1], 1)
    False
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (CONJ (NP (NNS oranges)) (NP (NNS bananas))))))")
    >>> fix_conj(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert t is not None

    # Check if the specified position is valid and corresponds to a 'CONJ' node
    if not isinstance(t[pos], Tree) or t[pos].label() != 'CONJ':
        return False

    new_children = []
    # Collect children, flattening nested conjunctions
    for child in t[pos]:
        if isinstance(child, Tree) and child.label() == 'CONJ':
            new_children.extend(child)  # Extend with children of the nested CONJ
        else:
            new_children.append(child)  # Append non-CONJ children

    # If no changes were made, return False
    if len(new_children) <= len(t[pos]):
        return False

    # Create a new tree for the flattened conjunction
    new_tree = Tree('CONJ', children=[Tree.convert(t) for t in new_children])

    # Replace the original 'CONJ' node with the new tree
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree))

    return True


def fix_all_coord(t: ParentedTree) -> None:
    """
    Fixes all coordination structures in a `ParentedTree`.

    This function iteratively applies `fix_coord` and `fix_conj` to the tree
    until no further modifications can be made. It ensures that the tree adheres
    to the correct syntactic structure for coordination and conjunctions.

    :param t: The `ParentedTree` in which coordination structures will be fixed.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_all_coord(t)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))

    >>> t2 = ParentedTree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges) (COORD (CCONJ and) (NP (NNS bananas))))))))")
    >>> fix_all_coord(t2)
    >>> print(t2.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert t is not None

    # Fix coordination
    coord_fixed = True
    while coord_fixed:
        coord_fixed = False
        for pos in t.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix coordination
            if fix_coord(t[pos[:-1]], pos[-1]):
                coord_fixed = True
                break

    # Fix conjunctions
    conj_fixed = True
    while conj_fixed:
        conj_fixed = False
        for pos in t.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix conjunctions
            if fix_conj(t[pos[:-1]], pos[-1]):
                conj_fixed = True
                break


def ins_ent(t: ParentedTree, tree_ent: TreeEntity) -> ParentedTree:
    """
    Inserts a tree entity into the appropriate position within a parented tree. The function modifies the tree
    structure to insert an entity at the correct level based on its positions and root position.

    :param t: A `ParentedTree` representing the syntactic tree.
    :param tree_ent: A `TreeEntity` containing the entity name and its positions in the tree.
    :return: The updated subtree where the entity was inserted.

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent1 = TreeEntity(name="person", positions=[(0, 0)])
    >>> tree_ent2 = TreeEntity(name="fruit", positions=[(1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent1)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (NP (NNS apples))))
    >>> ent_tree = ins_ent(t, tree_ent2)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (ENT::fruit apples)))

    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent = TreeEntity(name="xxx", positions=[(1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (ENT::xxx like apples))

    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice apples) (VP (VB like)))

    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like apples))
    >>> tree_ent = TreeEntity(name="yyy", positions=[(0, 2)])
    >>> ent_tree = ins_ent(t, tree_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like (ENT::yyy apples)))
    """
    assert t is not None

    # Determine the insertion point based on the positions of the entity
    anchor_pos = tree_ent.root_pos
    anchor_pos_len = len(anchor_pos)
    child_pos = tree_ent.positions[0]

    if sum(child_pos[anchor_pos_len + 1 :]) > 0:
        # Entity has children; attach to the common parent at the first child index + 1
        entity_index = child_pos[anchor_pos_len] + 1

    elif (
        t[tree_ent.root_pos].parent() is None
        or child_pos[anchor_pos_len] > 0
        or tree_ent.positions[-1][anchor_pos_len] < (len(t[tree_ent.root_pos]) - 1)
    ):
        # Attach to common parent at the correct index
        entity_index = child_pos[anchor_pos_len]

    else:
        # Attach to the grandparent at the common parent index
        entity_index = tree_ent.root_pos[-1]
        anchor_pos = tree_ent.root_pos[:-1]

        # Adjust anchor position upwards if necessary
        while len(t[anchor_pos]) == 1 and t[anchor_pos].parent():
            entity_index = anchor_pos[-1]
            anchor_pos = anchor_pos[:-1]

    # Collect and delete children from the original positions
    children = []
    for child_position in reversed(tree_ent.positions):
        children.append(t[child_position])
        del_elem(t[child_position[:-1]], child_position[-1])  # Delete child after conversion

    # Create a new tree node for the entity and insert it into the tree
    new_tree = Tree(NodeLabel(NodeType.ENT, tree_ent.name), children=reversed(children))
    ins_elem(t[anchor_pos], new_tree, entity_index)

    # Return the modified subtree where the entity was inserted
    return t[anchor_pos][entity_index]


def unnest_ent(t: ParentedTree, pos: int) -> None:
    """
    Un-nests an entity in a `ParentedTree` at the specified position `pos`.
    If the node at `pos` is labeled as an entity (ENT), the function converts
    the nested structure into a flat structure, creating a relationship (REL)
    between the entity and its nested entities.

    :param t: The `ParentedTree` in which the entity will be un-nested.
    :param pos: The index of the 'ENT' node to be processed.

    Example:
    >>> t = ParentedTree.fromstring('(S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))')
    >>> unnest_ent(t, 0)
    >>> print(t.pformat(margin=255))
    (S (REL (ENT::person Alice Bob Charlie) (nested (ENT::person Bob) (ENT::person Charlie))))
    """
    assert t is not None

    # Check if the specified position corresponds to an 'ENT' node
    if not has_type(t[pos], NodeType.ENT):
        return

    # Create the main entity tree and collect nested entities
    entity_tree = Tree(t[pos].label(), children=t[pos].leaves())

    # Collect nested entities and ensure they are only from the children of the current entity
    nested_entities = [child for child in t[pos] if has_type(child, NodeType.ENT)]
    nested_tree = Tree('nested', children=[Tree.convert(ne) for ne in nested_entities])

    # Construct a new relationship tree with the entity and its nested entities
    new_tree = Tree(NodeLabel(NodeType.REL), children=[entity_tree, nested_tree]) if nested_entities else entity_tree

    # Replace the original entity node with the new structure
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree))


def ins_rel(t: ParentedTree, tree_rel: TreeRel) -> None:
    assert t is not None


def enrich_tree(t: ParentedTree, sentence: str, entities: list[Entity], relations: list[Relation]) -> None:
    """
    Enriches a syntactic tree (`ParentedTree`) by inserting entities and relationships, and removing unused subtrees.

    The function processes a list of entities and relations, inserting them into the tree, unnesting entities as needed,
    and finally deleting any subtrees that are not part of the enriched structure.

    :param t: A `ParentedTree` representing the syntactic tree to enrich.
    :param sentence: The original sentence from which the tree is derived.
    :param entities: A list of `Entity` objects to be inserted into the tree.
    :param relations: A list of `Relation` objects representing the relationships between entities (currently not used).

    Example:
    >>> t = ParentedTree.fromstring("(S (NP Alice) (VP (VB likes) (NP (NNS apples) (CCONJ and) (NNS oranges))))")
    >>> e1 = Entity(name="person", start=0, end=5, id="E1")
    >>> e2 = Entity(name="fruit", start=12, end=18, id="E2")
    >>> e3 = Entity(name="fruit", start=23, end=30, id="E3")
    >>> enrich_tree(t, "Alice likes apples and oranges", [e1, e2, e3], [])
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (NP (ENT::fruit apples) (ENT::fruit oranges))))
    """
    assert t is not None
    assert sentence

    tokens = align_tokens(t.leaves(), sentence)

    # Insert entities into the tree by length (descending) to handle larger entities first
    computed_spans: set[TREE_POS] = set()
    entities_tree: list[ParentedTree] = []
    for entity in sorted(entities, key=len, reverse=True):
        entity_tokens = tuple(i for i, token in enumerate(tokens) if entity.start <= token[1] and token[0] < entity.end)

        # We do not support two entities at the same place
        if entity_tokens in computed_spans:
            continue

        tree_entity = TreeEntity(entity.name, [t.leaf_treeposition(i) for i in entity_tokens])
        entity_tree = ins_ent(t, tree_entity)
        entities_tree.append(entity_tree)
        computed_spans.add(entity_tokens)

    # Unnest any nested entities in reverse order (to avoid modifying parent indices during the process)
    for entity_tree in reversed(entities_tree):
        unnest_ent(entity_tree.parent(), entity_tree.parent_index())

    # Currently, the relations part is commented out, but can be enabled when relations are processed.
    for relation in relations:
        tree_rel = TreeRel((), (), relation.name)
        ins_rel(t, tree_rel)

    # Remove subtrees that have no specific entity or relation (i.e., generic subtrees)
    for subtree in list(t.subtrees(lambda x: x.height() == 2 and not has_type(x))):
        del_elem(subtree.parent(), subtree.parent_index())

"""
Generator of instances
"""

from collections.abc import Generator, Iterable

from architxt.model import NodeLabel, NodeType
from architxt.tree import Tree

GROUP_SCHEMA = dict[str, tuple[str, ...]]
REL_SCHEMA = dict[str, tuple[str, str]]


def gen_group(name: str, elements: tuple[str, ...]) -> Tree:
    """
    Generates a group tree structure with the given name and elements.
    :param name: The name of the group.
    :param elements: The elements to include in the group.
    :return: The generated group tree.

    Example:
    >>> group_tree = gen_group('Fruits', ('Apple', 'Banana', 'Cherry'))
    >>> print(group_tree.pformat(margin=255))
    (GROUP::Fruits (ENT::Apple ) (ENT::Banana ) (ENT::Cherry ))
    """
    label = NodeLabel(NodeType.GROUP, name)
    element_trees = [Tree(NodeLabel(NodeType.ENT, element), []) for element in elements]
    return Tree(label, element_trees)


def gen_relation(name: str, sub: str, obj: str, groups: GROUP_SCHEMA) -> Tree:
    """
    Generates a relation tree structure based on the given parameters.
    :param name: The name of the relationship.
    :param sub: The subject of the relationship.
    :param obj: The object of the relationship.
    :param groups: A dictionary containing group schemas.
    :return: The generated relation tree.

    Example:
    >>> groups = {'Fruits': ('Apple', 'Banana'), 'Colors': ('Red', 'Blue')}
    >>> relation_tree = gen_relation('Preference', 'Fruits', 'Colors', groups)
    >>> print(relation_tree.pformat(margin=255))
    (REL::Preference (GROUP::Fruits (ENT::Apple ) (ENT::Banana )) (GROUP::Colors (ENT::Red ) (ENT::Blue )))
    """
    label = NodeLabel(NodeType.REL, name)
    subject_tree = gen_group(sub, groups[sub])
    object_tree = gen_group(obj, groups[obj])
    return Tree(label, [subject_tree, object_tree])


def gen_collection(name: str, elements: Iterable[Tree]) -> Tree:
    """
    Generate a collection tree.
    :param name: The name of the collection.
    :param elements: The list of trees that make up the collection.
    :return: A tree representing the collection.

    Example:
    >>> from architxt.tree import Tree
    >>> elems = [Tree('Element1', []), Tree('Element2', [])]
    >>> collection_tree = gen_collection('Collection', elems)
    >>> print(collection_tree.pformat(margin=255))
    (COLL::Collection (Element1 ) (Element2 ))
    """
    label = NodeLabel(NodeType.COLL, name)
    return Tree(label, elements)


def gen_instance(
    groups: GROUP_SCHEMA, rels: REL_SCHEMA, *, size: int = 200, generate_collections: bool = True
) -> Generator[Tree, None, None]:
    """
    Generate a database instances as a tree based on the given groups and relations schema.
    :param groups: A dictionary containing group names as keys and elements as values.
    :param rels: A dictionary containing relation names as keys and tuples of sub and obj as values.
    :param size: An integer specifying the size of the generated trees.
    :param generate_collections: A boolean indicating whether to generate collections or not.
    :return: A tree representing the generated instance.
    """
    # Generate tree instances for each group
    for group_name, elements in groups.items():
        generated = (gen_group(group_name, elements) for _ in range(size))

        if generate_collections:
            yield gen_collection(group_name, generated)

        else:
            yield from generated

    # Generate tree instances for each relation
    for rel_name, (sub, obj) in rels.items():
        generated = (gen_relation(rel_name, sub, obj, groups) for _ in range(size))

        if generate_collections:
            yield gen_collection(rel_name, generated)

        else:
            yield from generated

"""
Generator of instances
"""

from collections.abc import Generator, Iterable

from architxt.schema import Schema
from architxt.tree import NodeLabel, NodeType, Tree

__all__ = ['gen_instance']

GROUP_SCHEMA = dict[str, tuple[str, ...]]
REL_SCHEMA = dict[str, tuple[str, str]]


def gen_group(schema: Schema, name: NodeLabel) -> Tree:
    """
    Generates a group tree structure with the given name and elements.
    :param schema: A schema to guide the tree structure.
    :param name: The name of the group.
    :return: The generated group tree.

    Example:
    >>> schema = Schema.from_description(groups={'Fruits': {'Apple', 'Banana', 'Cherry'}})
    >>> group_tree = gen_group(schema, NodeLabel(NodeType.GROUP, 'Fruits'))
    >>> print(group_tree.pformat(margin=255))
    (GROUP::Fruits (ENT::Apple data) (ENT::Banana data) (ENT::Cherry data))
    """
    return Tree(name, [Tree(element, ['data']) for element in sorted(schema.groups[name])])


def gen_relation(schema: Schema, name: NodeLabel) -> Tree:
    """
    Generates a relation tree structure based on the given parameters.
    :param schema: A schema to guide the tree structure.
    :param name: The name of the relationship.
    :return: The generated relation tree.

    Example:
    >>> schema = Schema.from_description(
    ...     groups={'Fruits': {'Apple', 'Banana'}, 'Colors': {'Red', 'Blue'}},
    ...     rels={'Preference': ('Fruits', 'Colors')}
    ... )
    >>> relation_tree = gen_relation(schema, NodeLabel(NodeType.REL, 'Preference'))
    >>> print(relation_tree.pformat(margin=255))
    (REL::Preference (GROUP::Colors (ENT::Blue data) (ENT::Red data)) (GROUP::Fruits (ENT::Apple data) (ENT::Banana data)))
    """
    sub, obj = schema.relations[name]
    subject_tree = gen_group(schema, sub)
    object_tree = gen_group(schema, obj)
    return Tree(name, [subject_tree, object_tree])


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


def gen_instance(schema: Schema, *, size: int = 200, generate_collections: bool = True) -> Generator[Tree, None, None]:
    """
    Generate a database instances as a tree based on the given groups and relations schema.
    :param schema: A schema to guide the tree structure.
    :param size: An integer specifying the size of the generated trees.
    :param generate_collections: A boolean indicating whether to generate collections or not.
    :return: A tree representing the generated instance.
    """
    # Generate tree instances for each group
    for group in schema.groups:
        generated = (gen_group(schema, group) for _ in range(size))

        if generate_collections:
            yield gen_collection(group.name, generated)

        else:
            yield from generated

    # Generate tree instances for each relation
    for relation in schema.relations:
        generated = (gen_relation(schema, relation) for _ in range(size))

        if generate_collections:
            yield gen_collection(relation.name, generated)

        else:
            yield from generated

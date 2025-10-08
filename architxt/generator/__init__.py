"""Generator of instances."""

from collections.abc import Generator, Iterable

from architxt.schema import Schema
from architxt.tree import NodeLabel, NodeType, Tree

__all__ = ['gen_instance']


def gen_group(schema: Schema, name: str) -> Tree:
    """
    Generate a group tree structure with the given name and elements.

    :param schema: A schema to guide the tree structure.
    :param name: The name of the group.
    :return: The generated group tree.

    >>> from architxt.schema import Group
    >>> group = Group(name='Fruits', entities={'Apple', 'Banana', 'Cherry'})
    >>> schema = Schema.from_description(groups={group})
    >>> group_tree = gen_group(schema, 'Fruits')
    >>> print(group_tree)
    (GROUP::Fruits (ENT::Apple data) (ENT::Banana data) (ENT::Cherry data))

    """
    group = next(group for group in schema.groups if group.name == name)
    entities = [Tree(NodeLabel(NodeType.ENT, ent), ['data']) for ent in sorted(group.entities)]

    return Tree(NodeLabel(NodeType.GROUP, group.name), entities)


def gen_relation(schema: Schema, name: str) -> Tree:
    """
    Generate a relation tree structure based on the given parameters.

    :param schema: A schema to guide the tree structure.
    :param name: The name of the relationship.
    :return: The generated relation tree.

    >>> from architxt.schema import Group, Relation
    >>> schema = Schema.from_description(
    ...     groups={
    ...         Group(name='Fruits', entities={'Apple', 'Banana'}),
    ...         Group(name='Colors', entities={'Red', 'Blue'}),
    ...     },
    ...     relations={Relation(name='Preference', left='Fruits', right='Colors')}
    ... )
    >>> relation_tree = gen_relation(schema, 'Preference')
    >>> print(relation_tree)
    (REL::Preference (GROUP::Fruits (ENT::Apple data) (ENT::Banana data)) (GROUP::Colors (ENT::Blue data) (ENT::Red data)))
    """
    rel = next(rel for rel in schema.relations if rel.name == name)
    subject_tree = gen_group(schema, rel.left)
    object_tree = gen_group(schema, rel.right)
    return Tree(NodeLabel(NodeType.REL, rel.name), [subject_tree, object_tree])


def gen_collection(name: str, elements: Iterable[Tree]) -> Tree:
    """
    Generate a collection tree.

    :param name: The name of the collection.
    :param elements: The list of trees that make up the collection.
    :return: A tree representing the collection.

    >>> from architxt.tree import Tree
    >>> elems = [Tree('Element1', []), Tree('Element2', [])]
    >>> collection_tree = gen_collection('Collection', elems)
    >>> print(collection_tree)
    (COLL::Collection (Element1 ) (Element2 ))
    """
    label = NodeLabel(NodeType.COLL, name)
    return Tree(label, elements)


def gen_instance(schema: Schema, *, size: int = 200, generate_collections: bool = True) -> Generator[Tree, None, None]:
    """
    Generate a database instance as a tree, based on the given groups and relations schema.

    :param schema: A schema to guide the tree structure.
    :param size: An integer specifying the size of the generated trees.
    :param generate_collections: A boolean indicating whether to generate collections or not.
    :return: A tree representing the generated instance.
    """
    # Generate tree instances for each group
    for group in schema.groups:
        generated = (gen_group(schema, group.name) for _ in range(size))

        if generate_collections:
            yield gen_collection(group.name, generated)

        else:
            yield from generated

    # Generate tree instances for each relation
    for relation in schema.relations:
        generated = (gen_relation(schema, relation.name) for _ in range(size))

        if generate_collections:
            yield gen_collection(relation.name, generated)

        else:
            yield from generated

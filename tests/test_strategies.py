"""Hypothesis strategies for property-based testing."""

import string
from itertools import combinations

from architxt.model import NodeLabel, NodeType
from architxt.schema import Schema
from architxt.tree import Tree, has_type
from hypothesis import given, note
from hypothesis import strategies as st

__all__ = ['tree_st']

LABEL_ST = st.text(string.ascii_uppercase, min_size=1, max_size=4)


@st.composite
def schema_st(
    draw: st.DrawFn,
    *,
    entities: set[str] | None = None,
    groups: dict[str, set[str]] | None = None,
    rels: dict[str, tuple[str, str]] | None = None,
    collections: bool = True,
) -> Schema:
    """
    Hypothesis strategy for generating `Schema` objects.

    :param draw: The Hypothesis draw function.
    :param entities: A set of predefined entities, or `None` to generate randomly.
    :param groups: A dictionary mapping group names to sets of entities, or `None` to generate randomly.
    :param rels: A dictionary mapping relation names to group pairs, or `None` to generate randomly.
    :param collections: If `True`, includes collections in the generated schema.
    :return: A `Schema` object generated based on the provided parameters.
    """

    if entities is None:
        entities = draw(st.sets(LABEL_ST, min_size=4, max_size=16))

    if groups is None:
        group_names = draw(st.lists(LABEL_ST, min_size=1, max_size=6, unique=True))
        groups = {
            group_name: draw(st.sets(st.sampled_from(list(entities)), min_size=2, max_size=min(5, len(entities))))
            for group_name in group_names
        }

    if rels is None and len(groups) >= 2:
        group_pairs = list(combinations(groups.keys(), 2))
        rels = {
            f'{group1}<->{group2}': (group1, group2)
            for group1, group2 in draw(st.lists(st.sampled_from(group_pairs), min_size=0, max_size=len(group_pairs)))
        }

    schema = Schema.from_description(groups=groups, rels=rels, collections=collections)
    note(f'== Generated Schema ==\n{schema.as_cfg()}\n============')

    return schema


@st.composite
def entity_tree_st(
    draw: st.DrawFn, *, schema: Schema | None = None, group_name: NodeLabel | None = None, name: str | None = None
) -> Tree:
    """
    Hypothesis strategy for generating an entity `Tree`.

    The entity can be drawn from a schema, restricted to a specific group, or randomly generated.

    :param draw: The Hypothesis draw function.
    :param schema: A schema to select entities from, or `None` for random generation.
    :param group_name: If provided, restricts entity selection to a specific group in the schema.
    :param name: If provided, use this as the entity name; otherwise, generates a random one.
    :return: An entity `Tree`.
    """
    if schema and (entities := list(schema.groups.get(group_name, schema.entities))):
        label = draw(st.sampled_from(entities))
    else:
        entity_name = name or draw(LABEL_ST)
        label = NodeLabel(NodeType.ENT, entity_name)

    return Tree(label, ['word'])


@st.composite
def group_tree_st(draw: st.DrawFn, *, schema: Schema | None = None, name: str | None = None) -> Tree:
    """
    Hypothesis strategy for generating a group `Tree`.

    The group can be drawn from a schema or randomly generated.

    :param draw: The Hypothesis draw function.
    :param schema: A schema to select groups and entities from, or `None` for random generation.
    :param name: If provided, use this as the group name; otherwise, generates a random one.
    :return: A group `Tree`.
    """
    if schema and schema.groups:
        if name and (label_name := NodeLabel(NodeType.GROUP, name)) in schema.groups:
            label = label_name

        else:
            groups = list(schema.groups.keys())
            label = draw(st.sampled_from(groups))

    else:
        group_name = name or draw(LABEL_ST)
        label = NodeLabel(NodeType.GROUP, group_name)

    entity_strategy = entity_tree_st(schema=schema, group_name=label)
    children = draw(st.lists(entity_strategy, min_size=2, unique_by=lambda tree: tree.label().name))

    return Tree(label, children)


@st.composite
def relation_tree_st(draw: st.DrawFn, *, schema: Schema | None = None, name: str | None = None) -> Tree:
    """
    Hypothesis strategy for generating a relation `Tree`.

    The relation can be drawn from a schema or randomly generated.

    :param draw: The Hypothesis draw function.
    :param schema: A schema to select relations and groups from, or `None` for random generation.
    :param name: If provided, use this as the relation name; otherwise, generates a random one.
    :return: A relation `Tree`.
    """
    if schema and schema.relations:
        if name and (label_name := NodeLabel(NodeType.REL, name)) in schema.relations:
            label = label_name

        else:
            relations = list(schema.relations.keys())
            label = draw(st.sampled_from(relations))

        group_1, group_2 = sorted(schema.relations[label], key=lambda x: x.name)
        children = draw(
            st.tuples(group_tree_st(schema=schema, name=group_1.name), group_tree_st(schema=schema, name=group_2.name))
        )

    else:
        relation_name = name or draw(LABEL_ST)
        label = NodeLabel(NodeType.REL, relation_name)
        children = draw(
            st.lists(group_tree_st(schema=schema), min_size=2, max_size=2, unique_by=lambda tree: tree.label().name)
        )

    return Tree(label, children)


@st.composite
def collection_tree_st(draw: st.DrawFn, *, schema: Schema | None = None) -> Tree:
    """
    Hypothesis strategy for generating a `Collection Tree`.

    The collections can be built from groups and relations in the given schema, or it can randomly generate them.

    :param draw: The Hypothesis draw function.
    :param schema: A schema to guide the tree generation, or `None` for random generation.
    :return: A collection `Tree`.
    """
    if schema is None:
        schema = draw(schema_st())

    if schema.relations and draw(st.booleans()):
        relations = list(schema.relations.keys())
        child_label = draw(st.sampled_from(relations))
        child_strategy = relation_tree_st(schema=schema, name=child_label.name)

    else:
        groups = list(schema.groups.keys())
        child_label = draw(st.sampled_from(groups))
        child_strategy = group_tree_st(schema=schema, name=child_label.name)

    label = NodeLabel(NodeType.COLL, child_label.name)
    children = draw(st.lists(child_strategy, min_size=2))
    return Tree(label, children)


@st.composite
def tree_st(draw: st.DrawFn, *, schema: Schema | None = None, has_parent: bool | None = None) -> Tree:
    """
    Hypothesis strategy for generating a general-purpose tree structure.

    The tree may be composed of entities, groups, relations, or collections, based on the provided schema.
    This strategy supports recursive generation to produce deeply nested tree structures.

    :param draw: The Hypothesis draw function.
    :param schema: A schema to guide the tree structure, or `None` for random generation.
    :param has_parent: If `True`, ensures the tree has a parent; otherwise, generates a standalone tree.
    :return: An instance `Tree`.
    """
    if not schema:
        schema = draw(schema_st())

    if has_parent is None:
        has_parent = draw(st.booleans())

    base_strategy = (
        st.one_of(
            entity_tree_st(schema=schema),
            group_tree_st(schema=schema),
            relation_tree_st(schema=schema),
            collection_tree_st(schema=schema),
        )
        if schema.relations
        else st.one_of(
            entity_tree_st(schema=schema),
            group_tree_st(schema=schema),
            collection_tree_st(schema=schema),
        )
    )

    tree = draw(
        st.recursive(
            base_strategy,
            lambda children: st.builds(Tree, st.just(''), st.lists(children, min_size=1, max_size=10)),
            max_leaves=6,
        )
    )

    if has_type(tree):
        tree = Tree('ROOT', [tree])
    else:
        tree.set_label('ROOT')

    note(f'== Generated Tree ==\n{tree.pformat()}\n============')
    return draw(st.sampled_from(list(tree.subtrees()))) if has_parent else tree


@given(schema=schema_st())
def test_schema_generation(schema: Schema):
    """
    Property-based test for verifying generated schemas.

    Ensures that schemas created by `schema_st` are valid, according to the meta-grammar.

    :param schema: A `Schema` object generated by `schema_st`.
    """
    assert schema.verify()


@given(schema=schema_st(), data=st.data())
def test_entity_tree_generation(schema: Schema, data: st.DataObject):
    """
    Property-based test for verifying that `entity_tree_st` generates only entities from the schema.

    :param schema: A `Schema` object to guide tree generation.
    :param data: An instance of data needed to draw entity trees.
    """
    entity_tree = data.draw(entity_tree_st(schema=schema))
    if schema.entities:
        assert entity_tree.label() in schema.entities


@given(schema=schema_st(), data=st.data())
def test_group_tree_generation(schema: Schema, data: st.DataObject):
    """
    Property-based test for verifying that `group_tree_st` generates only groups and entities from the schema.

    :param schema: A `Schema` object to guide tree generation.
    :param data: An instance of data needed to draw group trees.
    """
    group_tree = data.draw(group_tree_st(schema=schema))
    if schema.groups:
        # Ensure the group tree's label is in the schema's groups
        assert group_tree.label() in schema.groups
        # Ensure the group tree's children are entities defined in the schema for that group
        group_entities = {child.label() for child in group_tree}
        assert group_entities.issubset(schema.groups[group_tree.label()])


@given(tree=tree_st(has_parent=False))
def test_instance_generation(tree: Tree):
    """
    Property-based test for verifying generated trees.

    Ensures that a tree structure generated by `tree_st` can be serialized into a schema
    and that the resulting schema is valid, according to the meta-grammar.

    :param tree: A `Tree` object generated by `tree_st`.
    """
    schema = Schema.from_forest([tree], keep_unlabelled=False)
    assert schema.verify()

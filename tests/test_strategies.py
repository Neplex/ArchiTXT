"""Hypothesis strategies for property-based testing."""

import string
from itertools import combinations

from architxt.schema import Group, Relation, RelationOrientation, Schema
from architxt.tree import NodeLabel, NodeType, Tree, has_type
from hypothesis import given, note
from hypothesis import strategies as st

__all__ = ['tree_st']

LABEL_ST = st.text(string.ascii_uppercase, min_size=1, max_size=4)


@st.composite
def schema_st(
    draw: st.DrawFn,
    *,
    entities: set[str] | None = None,
    groups: set[Group] | None = None,
    relations: set[Relation] | None = None,
    collections: bool = True,
) -> Schema:
    """
    Hypothesis strategy for generating `Schema` objects.

    :param draw: The Hypothesis draw function.
    :param entities: A set of predefined entities, or `None` to generate randomly.
    :param groups: A dictionary mapping group names to sets of entities, or `None` to generate randomly.
    :param relations: A dictionary mapping relation names to group pairs, or `None` to generate randomly.
    :param collections: If `True`, includes collections in the generated schema.
    :return: A `Schema` object generated based on the provided parameters.
    """
    if entities is None:
        entities = draw(st.sets(LABEL_ST, min_size=4, max_size=16))

    if groups is None:
        group_names = draw(st.lists(LABEL_ST, min_size=1, max_size=6, unique=True))
        groups = {
            Group(
                name=group_name,
                entities=draw(st.sets(st.sampled_from(list(entities)), min_size=2, max_size=min(5, len(entities)))),
            )
            for group_name in group_names
        }

    if relations is None and len(groups) >= 2:
        group_pairs = list(combinations(groups, 2))
        relations = {
            Relation(
                name=f'{group1.name}<->{group2.name}',
                left=group1.name,
                right=group2.name,
                orientation=draw(st.sampled_from(RelationOrientation)),
            )
            for group1, group2 in draw(st.lists(st.sampled_from(group_pairs), min_size=0, max_size=len(group_pairs)))
        }

    schema = Schema.from_description(groups=groups, relations=relations, collections=collections)
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
    if schema and schema.entities:
        entities = next((group.entities for group in schema.groups if group.name == group_name), schema.entities)
        entity_name = draw(st.sampled_from(list(entities)))

    else:
        entity_name = name or draw(LABEL_ST)

    label = NodeLabel(NodeType.ENT, entity_name)
    words = draw(
        st.lists(
            st.text(string.digits + string.ascii_letters + string.punctuation, min_size=1, max_size=6),
            min_size=1,
            max_size=3,
        )
    )

    return Tree(label, words)


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
        # pick the named group if it exists, otherwise sample one
        group = next((group for group in schema.groups if name and group.name == name), None) or draw(
            st.sampled_from(list(schema.groups))
        )
        group_name = group.name

    else:
        group_name = name or draw(LABEL_ST)

    entity_strategy = entity_tree_st(schema=schema, group_name=group_name)
    children = draw(st.lists(entity_strategy, min_size=2, unique_by=lambda tree: tree.label.name))

    label = NodeLabel(NodeType.GROUP, group_name)
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
        # pick the named relation if it exists, otherwise sample one
        relation = next((rel for rel in schema.relations if name and rel.name == name), None) or draw(
            st.sampled_from(list(schema.relations))
        )

        relation_name = relation.name
        children = draw(
            st.tuples(
                group_tree_st(schema=schema, name=relation.left),
                group_tree_st(schema=schema, name=relation.right),
            )
        )

    else:
        relation_name = name or draw(LABEL_ST)
        children = draw(
            st.lists(group_tree_st(schema=schema), min_size=2, max_size=2, unique_by=lambda tree: tree.label.name)
        )

    label = NodeLabel(NodeType.REL, relation_name)
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
        relation = draw(st.sampled_from(list(schema.relations)))
        child_name = relation.name
        child_strategy = relation_tree_st(schema=schema, name=child_name)

    else:
        group = draw(st.sampled_from(list(schema.groups)))
        child_name = group.name
        child_strategy = group_tree_st(schema=schema, name=child_name)

    label = NodeLabel(NodeType.COLL, child_name)
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
        tree.label = 'ROOT'

    note(f'== Generated Tree ==\n{tree.pformat()}\n============')
    return draw(st.sampled_from(list(tree.subtrees()))) if has_parent else tree


@given(schema=schema_st())
def test_schema_generation(schema: Schema) -> None:
    """
    Property-based test for verifying generated schemas.

    Ensures that schemas created by `schema_st` are valid, according to the meta-grammar.

    :param schema: A `Schema` object generated by `schema_st`.
    """
    assert schema.verify()


@given(schema=schema_st(), data=st.data())
def test_entity_tree_generation(schema: Schema, data: st.DataObject) -> None:
    """
    Property-based test for verifying that `entity_tree_st` generates only entities from the schema.

    :param schema: A `Schema` object to guide tree generation.
    :param data: An instance of data needed to draw entity trees.
    """
    entity_tree = data.draw(entity_tree_st(schema=schema))
    if schema.entities:
        assert entity_tree.label.name in schema.entities


@given(schema=schema_st(), data=st.data())
def test_group_tree_generation(schema: Schema, data: st.DataObject) -> None:
    """
    Property-based test for verifying that `group_tree_st` generates only groups and entities from the schema.

    :param schema: A `Schema` object to guide tree generation.
    :param data: An instance of data needed to draw group trees.
    """
    group_tree = data.draw(group_tree_st(schema=schema))
    if schema.groups:
        # Ensure the group tree's label is in the schema's groups
        all_groups_name = [group.name for group in schema.groups]
        assert group_tree.label.name in all_groups_name

        # Ensure the group tree's children are entities defined in the schema for that group
        sch_entities = next((group.entities for group in schema.groups if group.name == group_tree.label.name), set())
        group_entities = {child.label.name for child in group_tree}
        assert group_entities.issubset(sch_entities)


@given(tree=tree_st(has_parent=False))
def test_instance_generation(tree: Tree) -> None:
    """
    Property-based test for verifying generated trees.

    Ensures that a tree structure generated by `tree_st` can be serialized into a schema
    and that the resulting schema is valid, according to the meta-grammar.

    :param tree: A `Tree` object generated by `tree_st`.
    """
    schema = Schema.from_forest([tree], keep_unlabelled=False)
    assert schema.verify()

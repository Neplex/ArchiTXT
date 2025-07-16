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
    min_entities: int = 4,
    max_entities: int = 10,
    groups: set[Group] | None = None,
    max_group_count: int = 5,
    min_group_size: int = 2,
    max_group_size: int = 5,
    relations: set[Relation] | None = None,
    max_relations_count: int = 4,
    collections: bool = True,
) -> Schema:
    """
    Hypothesis strategy for generating `Schema` objects.

    :param draw: The Hypothesis's `draw` function.

    :param entities:  A set of predefined entity labels, or `None` to generate randomly.
    :param min_entities: Minimum number of distinct entities to generate when `entities` is `None`.
    :param max_entities: Maximum number of distinct entities to generate when `entities` is `None`.
    :param groups: A set of predefined `Group` objects, or `None` to generate randomly.
    :param max_group_count: Maximum number of groups to generate when `groups` is `None`.
    :param min_group_size: Minimum number of entities each group must contain (applies when `groups` is `None`).
    :param max_group_size: Maximum number of entities each group can contain (applies when `groups` is `None`).
    :param relations: A set of predefined `Relation` objects, or `None` to generate randomly.
    :param max_relations_count: Maximum number of relations to generate when `relations` is `None`.
        Relations will only be generated if there are at least two groups.
    :param collections: If `True`, includes collections in the generated schema.
    :return: A `Schema` object assembled from the drawn entities, groups, and relations.
    """
    if entities is None:
        entity_list = draw(st.lists(LABEL_ST, min_size=min_entities, max_size=max_entities, unique=True))
    else:
        entity_list = list(entities)

    if groups is None:
        min_size = max(min_group_size, len(entity_list))
        max_size = max(min_size, min(max_group_size, len(entity_list)))
        group_names = draw(st.lists(LABEL_ST, min_size=1, max_size=max_group_count, unique=True))
        groups = {
            Group(
                name=group_name,
                entities=draw(st.sets(st.sampled_from(entity_list), min_size=min_size, max_size=max_size)),
            )
            for group_name in group_names
        }

    if relations is None and max_relations_count and len(groups) >= 2:
        group_pairs = list(combinations(groups, 2))
        max_size = min(max_relations_count, len(group_pairs))
        relations = {
            Relation(
                name=f'{left.name}<->{right.name}',
                left=left.name,
                right=right.name,
                orientation=draw(st.sampled_from(RelationOrientation)),
            )
            for groups in draw(st.lists(st.sampled_from(group_pairs), min_size=0, max_size=max_size))
            for left, right in [sorted(groups, key=lambda g: g.name)]
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

    :param draw: The Hypothesis's `draw` function.
    :param schema: A schema to select entities from, or `None` for random generation.
    :param group_name: If provided, restricts entity selection to a specific group in the schema.
    :param name: If provided, use this as the entity name; otherwise, generates a random one.
    :return: An entity `Tree`.
    """
    if not name and schema and schema.entities:
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

    :param draw: The Hypothesis's `draw` function.
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
        entity_list = list(group.entities)

    else:
        group_name = name or draw(LABEL_ST)
        entity_list = list(schema.entities) if schema else draw(st.lists(LABEL_ST, min_size=1, max_size=8, unique=True))

    entities = draw(st.sets(st.sampled_from(entity_list), min_size=1))
    children = [draw(entity_tree_st(schema=schema, name=n)) for n in entities]

    label = NodeLabel(NodeType.GROUP, group_name)
    return Tree(label, children)


@st.composite
def relation_tree_st(draw: st.DrawFn, *, schema: Schema | None = None, name: str | None = None) -> Tree:
    """
    Hypothesis strategy for generating a relation `Tree`.

    The relation can be drawn from a schema or randomly generated.

    :param draw: The Hypothesis's `draw` function.
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

    :param draw: The Hypothesis's `draw` function.
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

    :param draw: The Hypothesis's `draw` function.
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
            lambda children: st.builds(Tree, st.just(''), st.lists(children, min_size=1, max_size=6)),
            max_leaves=4,
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


@given(data=st.data())
def test_entity_tree_generation(data: st.DataObject) -> None:
    """Property-based test for verifying that `entity_tree_st` generates only entities from the schema."""
    schema = data.draw(schema_st())
    entity_tree = data.draw(entity_tree_st(schema=schema))

    assert entity_tree.label.type == NodeType.ENT
    assert entity_tree.label.name in schema.entities


@given(data=st.data())
def test_group_tree_generation(data: st.DataObject) -> None:
    """Property-based test for verifying that `group_tree_st` generates only groups and entities from the schema."""
    schema = data.draw(schema_st() | st.none())
    group_tree = data.draw(group_tree_st(schema=schema))

    # Check that it is a group
    assert group_tree.label.type == NodeType.GROUP
    assert len(group_tree) > 0

    # Ensure no duplicate entities inside the group
    group_entities = [child.label.name for child in group_tree]
    assert len(group_entities) == len(set(group_entities))

    if schema:
        # Ensure the group tree's label is in the schema's groups
        all_groups_name = {group.name for group in schema.groups}
        assert group_tree.label.name in all_groups_name

        # Ensure the group tree's children are entities defined in the schema for that group
        sch_entities = next(group.entities for group in schema.groups if group.name == group_tree.label.name)
        assert sch_entities.issuperset(group_entities)


@given(data=st.data())
def test_relation_tree_generation(data: st.DataObject) -> None:
    """Property-based test to verify that `relation_tree_st` generates correct relations."""
    schema = data.draw(schema_st() | st.none())
    rel_tree = data.draw(relation_tree_st(schema=schema))

    # Check that it is a relation
    assert rel_tree.label.type == NodeType.REL
    assert len(rel_tree) == 2

    left_name, right_name = rel_tree[0].label.name, rel_tree[1].label.name
    assert left_name != right_name

    if schema and schema.relations:
        # Ensure relation is defined in schema
        relation_names = {rel.name for rel in schema.relations}
        assert rel_tree.label.name in relation_names

        # Ensure the relation tree's children are groups defined in the schema for that relation
        relation = next(rel for rel in schema.relations if rel.name == rel_tree.label.name)
        assert {left_name, right_name} == {relation.left, relation.right}


@given(data=st.data())
def test_collection_tree_generation(data: st.DataObject) -> None:
    """Property-based test to verify that `collection_tree_st` generates correct collection trees."""
    schema = data.draw(schema_st() | st.none())
    coll_tree = data.draw(collection_tree_st(schema=schema))

    # Check it's a collection node
    assert coll_tree.label.type == NodeType.COLL
    assert len(coll_tree) >= 2

    child_name = coll_tree[0].label.name
    child_type = coll_tree[0].label.type

    # Only groups or relations can be part of a collection
    assert child_type in {NodeType.GROUP, NodeType.REL}

    # All children should have the same label name and type
    assert all(child.label.name == child_name for child in coll_tree)
    assert all(child.label.type == child_type for child in coll_tree)

    if schema:
        group_names = {group.name for group in schema.groups}
        relation_names = {rel.name for rel in schema.relations}
        valid_names = group_names | relation_names

        # The label must exist in the schema
        assert child_name in valid_names

        # Enforce correct type-name consistency
        if child_name in group_names:
            assert child_type == NodeType.GROUP

        else:
            assert child_type == NodeType.REL


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

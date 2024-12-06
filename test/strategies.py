"""Hypothesis strategies for property-based testing."""

import string

from architxt.model import NodeLabel, NodeType
from architxt.tree import Tree
from hypothesis import strategies as st
from hypothesis.strategies import composite

__all__ = ['tree_st']


@composite
def node_label_st(draw: st.DrawFn, *, label_type: NodeType | None = None, label: str | None = None) -> NodeLabel | str:
    """
    Generates a NodeLabel with an optional type and label.

    :param draw: The hypothesis draw function.
    :param label_type: The type of the node to generate.
    :param label: The label of the node. If None, it will be randomly generated.
    """
    label = label if label is not None else draw(st.text(min_size=0, max_size=5, alphabet=string.ascii_letters))

    if label_type:
        return NodeLabel(label_type, label)

    return label


@composite
def entity_tree_st(draw: st.DrawFn, *, name: str | None = None) -> Tree:
    """
    Generates an Entity Tree with a random or specified name.

    :param draw: The hypothesis draw function.
    :param name: The name of the entity. If None, it will be randomly generated.
    """
    label = node_label_st(label_type=NodeType.ENT, label=name)
    children = st.lists(st.just("word"), min_size=1, max_size=5)
    return draw(st.builds(Tree, label, children))


@composite
def group_tree_st(draw: st.DrawFn, *, name: str | None = None) -> Tree:
    """
    Generates a Group Tree containing multiple Entity Trees.

    :param draw: The hypothesis draw function.
    :param name: The name of the group. If None, it will be randomly generated.
    """
    label = node_label_st(label_type=NodeType.GROUP, label=name)
    children = st.lists(
        entity_tree_st(),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x.label().name,
    )
    return draw(st.builds(Tree, label, children))


@composite
def relation_tree_st(draw: st.DrawFn, *, name: str | None = None) -> Tree:
    """
    Generates a Relation Tree containing multiple Group Trees.

    :param draw: The hypothesis draw function.
    :param name: The name of the relation. If None, it will be randomly generated.
    """
    label = node_label_st(label_type=NodeType.REL, label=name)
    children = st.lists(
        group_tree_st(),
        min_size=2,
        max_size=2,
        unique_by=lambda x: x.label().name,
    )
    return draw(st.builds(Tree, label, children))


@composite
def collection_tree_st(draw: st.DrawFn, *, name: str | None = None) -> Tree:
    """
    Generates a Collection Tree containing Group or Relation Trees.

    :param draw: The hypothesis draw function.
    :param name: The name of the collection. If None, it will be randomly generated.
    """
    name = name if name is not None else draw(st.text(min_size=0, max_size=5, alphabet=string.ascii_letters))
    label = node_label_st(label_type=NodeType.COLL, label=name)
    children = st.lists(
        st.one_of(group_tree_st(name=name), relation_tree_st(name=name)),
        min_size=1,
        max_size=5,
    )
    return draw(st.builds(Tree, label, children))


@composite
def tree_st(draw: st.DrawFn, *, has_parent: bool | None = None) -> Tree:
    """
    Generates a tuple containing a tree and one of its subtrees.

    :param draw: The hypothesis draw function.
    :param has_parent: Controls if the subtree should have a parent.
                       True = subtree must have a parent,
                       False = subtree must not have a parent,
                       None = randomly decided.
    """
    base_strategy = st.one_of(
        entity_tree_st(),
        group_tree_st(),
        relation_tree_st(),
        collection_tree_st(),
    )

    # Generate a recursive tree structure
    tree = draw(
        st.recursive(
            base_strategy,
            lambda children: st.builds(Tree, node_label_st(), st.lists(children, min_size=1, max_size=5)),
            max_leaves=5,
        )
    )

    # Handle `has_parent` flag
    has_parent = has_parent if has_parent is not None else draw(st.booleans())
    if not has_parent:
        return tree

    return draw(st.sampled_from(tree.subtrees()))

from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import FindSubGroupsOperation
from architxt.tree import Tree

from . import create_test_clusterer


def test_find_subgroups_no_simplify() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::X 1) (ENT::Y 2))')],
        },
        tau=0.8,
        min_support=0,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert not has_simplified
    assert str(tree) == '(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))'


def test_find_subgroups_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))'


def test_find_subgroups_simple_group() -> None:
    tree = Tree.fromstring('(SENT (GROUP::1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = create_test_clusterer(
        {
            '2': [
                Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),
                Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),
            ],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    tree[0].label = 'XXX'

    assert has_simplified
    assert str(tree) == '(SENT (XXX (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))'


def test_find_subgroups_largest() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4)))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3))')],
            '3': [Tree.fromstring('(GROUP::3 (ENT::A 1) (ENT::B 2))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (ENT::D 4)))'


def test_find_subgroups_multi() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4) (ENT::E 5)))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            '3': [Tree.fromstring('(GROUP::3 (ENT::D 4) (ENT::E 5))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))'


def test_find_subgroups_root() -> None:
    tree = Tree.fromstring('(SENT (ENT::A 1) (ENT::B 2) (ENT::C 3))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3))'

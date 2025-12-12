from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import MergeGroupsOperation
from architxt.tree import Tree

from . import create_test_clusterer


def test_merge_groups_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (GROUP::3 (ENT::C 3) (ENT::D 4))))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = MergeGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))))'


def test_merge_groups_extend() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))')

    clusterer = create_test_clusterer(
        {
            '2': [Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            '4': [Tree.fromstring('(GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3))')],
        },
        tau=0.8,
        min_support=0,
        metric=jaccard,
    )

    operation = MergeGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (GROUP::3 (ENT::D 4) (ENT::E 5))))'

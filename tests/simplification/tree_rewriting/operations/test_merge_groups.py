from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import MergeGroupsOperation
from architxt.tree import Tree


def test_merge_groups_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (GROUP::3 (ENT::C 3) (ENT::D 4))))')

    operation = MergeGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))))')


def test_merge_groups_extend() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))')

    operation = MergeGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
            (Tree.fromstring('(GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (1 (GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (GROUP::3 (ENT::D 4) (ENT::E 5))))'
    )

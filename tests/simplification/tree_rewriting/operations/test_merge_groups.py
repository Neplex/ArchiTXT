from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import merge_groups
from architxt.tree import Tree


def test_merge_groups_simple():
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (GROUP::3 (ENT::C 3) (ENT::D 4))))')

    equiv_subtrees = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))'),),
    }

    tree, has_simplified = merge_groups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))))')


def test_merge_groups_extend():
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))')

    equiv_subtrees = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
        (Tree.fromstring('(GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3))'),),
    }

    tree, has_simplified = merge_groups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (1 (GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (GROUP::3 (ENT::D 4) (ENT::E 5))))'
    )

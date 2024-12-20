from architxt.similarity import TREE_CLUSTER, jaccard
from architxt.simplification.tree_rewriting.operations import find_subgroups
from architxt.tree import Tree


def test_find_subgroups_no_simplify():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    equiv_subtrees: TREE_CLUSTER = {
        (Tree.fromstring('(GROUP::2 (ENT::X 1) (ENT::Y 2))'),),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')


def test_find_subgroups_simple():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    equiv_subtrees: TREE_CLUSTER = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))')


def test_find_subgroups_simple_group():
    tree = Tree.fromstring('(SENT (GROUP::1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    equiv_subtrees: TREE_CLUSTER = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'), Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(SENT ( (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))')


def test_find_subgroups_largest():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4)))')

    equiv_subtrees = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3))'),),
        (Tree.fromstring('(GROUP::3 (ENT::A 1) (ENT::B 2))'),),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (ENT::D 4)))')


def test_find_subgroups_multi():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4) (ENT::E 5)))')

    equiv_subtrees = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
        (Tree.fromstring('(GROUP::3 (ENT::D 4) (ENT::E 5))'),),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))'
    )


def test_find_subgroups_root():
    tree = Tree.fromstring('(SENT (ENT::A 1) (ENT::B 2) (ENT::C 3))')

    equiv_subtrees: TREE_CLUSTER = {
        (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
    }

    tree, has_simplified = find_subgroups(tree, equiv_subtrees, 0.8, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3))')

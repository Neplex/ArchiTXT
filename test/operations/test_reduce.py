from architxt.operations import reduce_bottom, reduce_top
from architxt.similarity import jaccard
from architxt.tree import Tree


def test_reduce_bottom_simple():
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    tree, has_reduced = reduce_bottom(tree, set(), 0.7, 0, jaccard)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_bottom_nested():
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    tree, has_reduced = reduce_bottom(tree, set(), 0.7, 0, jaccard)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))')


def test_reduce_bottom_no_reduction():
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    tree, has_reduced = reduce_bottom(tree, set(), 0.7, 0, jaccard)

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')


def test_reduce_top_simple():
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    tree, has_reduced = reduce_top(tree, set(), 0.7, 0, jaccard)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_top_nested():
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    tree, has_reduced = reduce_top(tree, set(), 0.7, 0, jaccard)

    print(tree.pformat())
    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc)))')


def test_reduce_top_no_reduction():
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    tree, has_reduced = reduce_top(tree, set(), 0.7, 0, jaccard)

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

from architxt.similarity import TreeClusterer
from architxt.simplification.tree_rewriting.operations import ReduceBottomOperation, ReduceTopOperation
from architxt.tree import Tree


def test_reduce_bottom_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    operation = ReduceBottomOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_bottom_nested() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    operation = ReduceBottomOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))')


def test_reduce_bottom_no_reduction() -> None:
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    operation = ReduceBottomOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')


def test_reduce_top_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    operation = ReduceTopOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_top_nested() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    operation = ReduceTopOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc)))')


def test_reduce_top_no_reduction() -> None:
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    operation = ReduceTopOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_reduced = operation.apply(tree)

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import ReduceBottomOperation, ReduceTopOperation
from architxt.tree import Tree


def test_reduce_bottom_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    operation = ReduceBottomOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_bottom_nested() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    operation = ReduceBottomOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (1 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))')


def test_reduce_bottom_no_reduction() -> None:
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    operation = ReduceBottomOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')


def test_reduce_top_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb))))')

    operation = ReduceTopOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)))')


def test_reduce_top_nested() -> None:
    tree = Tree.fromstring('(SENT (1 (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc))))')

    operation = ReduceTopOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert has_reduced
    assert tree == Tree.fromstring('(SENT (2 (ENT::A aaa) (ENT::B bbb)) (3 (ENT::C ccc)))')


def test_reduce_top_no_reduction() -> None:
    tree = Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

    operation = ReduceTopOperation(tau=0.7, min_support=0, metric=jaccard)
    tree, has_reduced = operation.apply(tree, equiv_subtrees=set())

    assert not has_reduced
    assert tree == Tree.fromstring('(SENT (ENT::A aaa) (ENT::B bbb))')

from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import FindSubGroupsOperation
from architxt.tree import Tree


def test_find_subgroups_no_simplify():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::X 1) (ENT::Y 2))'),),
        },
    )

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')


def test_find_subgroups_simple():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))')


def test_find_subgroups_simple_group():
    tree = Tree.fromstring('(SENT (GROUP::1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'), Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring('(SENT ( (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))')


def test_find_subgroups_largest():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4)))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3))'),),
            (Tree.fromstring('(GROUP::3 (ENT::A 1) (ENT::B 2))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (ENT::D 4)))')


def test_find_subgroups_multi():
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4) (ENT::E 5)))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
            (Tree.fromstring('(GROUP::3 (ENT::D 4) (ENT::E 5))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))'
    )


def test_find_subgroups_root():
    tree = Tree.fromstring('(SENT (ENT::A 1) (ENT::B 2) (ENT::C 3))')

    operation = FindSubGroupsOperation(tau=0.8, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(
        tree,
        equiv_subtrees={
            (Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),),
        },
    )

    assert has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3))')

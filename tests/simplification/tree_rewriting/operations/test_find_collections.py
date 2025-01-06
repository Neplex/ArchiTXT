from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import FindCollectionsOperation
from architxt.tree import Tree


def test_find_collections_simple():
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    operation = FindCollectionsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring('(COLL::A (GROUP::A x) (GROUP::A y))')


def test_find_collections_multi():
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (GROUP::A 2) (GROUP::B 3) (GROUP::B 4) (GROUP::A 5) (ENT 6) (GROUP::C 7) (GROUP::C 8))'
    )

    operation = FindCollectionsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 5)) (COLL::B (GROUP::B 3) (GROUP::B 4)) (ENT 6) (COLL::C (GROUP::C 7) (GROUP::C 8)))'
    )


def test_find_collections_merge():
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (COLL::A (GROUP::A 2) (GROUP::A 3)) (GROUP::A 4) (COLL::A (GROUP::A 5) (GROUP::A 6)))'
    )

    operation = FindCollectionsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring(
        '(COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 3) (GROUP::A 4) (GROUP::A 5) (GROUP::A 6))'
    )


def test_find_collections_naming_only():
    operation = FindCollectionsOperation(tau=0.0, min_support=0, metric=jaccard, naming_only=True)

    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring('(COLL::A (GROUP::A x) (GROUP::A y))')

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))')

    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))')

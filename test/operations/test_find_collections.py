from architxt.operations import find_collections
from architxt.similarity import jaccard
from architxt.tree import Tree


def test_find_collections_simple():
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    tree, has_simplified = find_collections(tree, set(), 0, 0, jaccard)

    assert not has_simplified
    assert tree == Tree.fromstring('(COLL::A (GROUP::A x) (GROUP::A y))')


def test_find_collections_multi():
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (GROUP::A 2) (GROUP::B 3) (GROUP::B 4) (GROUP::A 5) (ENT 6) (GROUP::C 7) (GROUP::C 8))'
    )

    tree, has_simplified = find_collections(tree, set(), 0, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 5)) (COLL::B (GROUP::B 3) (GROUP::B 4)) (ENT 6) (COLL::C (GROUP::C 7) (GROUP::C 8)))'
    )


def test_find_collections_merge():
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (COLL::A (GROUP::A 2) (GROUP::A 3)) (GROUP::A 4) (COLL::A (GROUP::A 5) (GROUP::A 6)))'
    )

    tree, has_simplified = find_collections(tree, set(), 0, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 3) (GROUP::A 4) (GROUP::A 5) (GROUP::A 6))'
    )


def test_find_collections_naming_only():
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    tree, has_simplified = find_collections(tree, set(), 0, 0, jaccard, naming_only=True)

    assert not has_simplified
    assert tree == Tree.fromstring('(COLL::A (GROUP::A x) (GROUP::A y))')

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))')

    tree, has_simplified = find_collections(tree, set(), 0, 0, jaccard, naming_only=True)

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))')

from architxt.similarity import TreeClusterer
from architxt.simplification.tree_rewriting.operations import FindCollectionsOperation
from architxt.tree import Tree


def test_find_collections_simple() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    operation = FindCollectionsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(COLL::A (GROUP::A x) (GROUP::A y))'


def test_find_collections_multi() -> None:
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (GROUP::A 2) (GROUP::B 3) (GROUP::B 4) (GROUP::A 5) (ENT 6) (GROUP::C 7) (GROUP::C 8))'
    )

    operation = FindCollectionsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert (
        str(tree)
        == '(SENT (COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 5)) (COLL::B (GROUP::B 3) (GROUP::B 4)) (ENT 6) (COLL::C (GROUP::C 7) (GROUP::C 8)))'
    )


def test_find_collections_merge() -> None:
    tree = Tree.fromstring(
        '(SENT (GROUP::A 1) (COLL::A (GROUP::A 2) (GROUP::A 3)) (GROUP::A 4) (COLL::A (GROUP::A 5) (GROUP::A 6)))'
    )

    operation = FindCollectionsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(COLL::A (GROUP::A 1) (GROUP::A 2) (GROUP::A 3) (GROUP::A 4) (GROUP::A 5) (GROUP::A 6))'


def test_find_collections_naming_only() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y))')

    operation = FindCollectionsOperation(tree_clusterer=TreeClusterer(), min_support=0, naming_only=True)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(COLL::A (GROUP::A x) (GROUP::A y))'

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))')

    has_simplified = operation.apply(tree)

    assert not has_simplified
    assert str(tree) == '(SENT (GROUP::A x) (GROUP::A y) (GROUP::B z))'

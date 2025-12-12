from architxt.similarity import TreeClusterer
from architxt.simplification.tree_rewriting.operations import FindRelationsOperation
from architxt.tree import Tree


def test_find_relations_simple() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    operation = FindRelationsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(REL::A<->B (GROUP::A x) (GROUP::B y))'


def test_find_relations_collection() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    operation = FindRelationsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (REL::A<->B (GROUP::A x) (GROUP::B 1)) (REL::A<->B (GROUP::A x) (GROUP::B 2)))'


def test_find_relations_collection_same_group() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))')

    operation = FindRelationsOperation(tree_clusterer=TreeClusterer(), min_support=0)
    has_simplified = operation.apply(tree)

    assert not has_simplified
    assert str(tree) == '(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))'


def test_find_relations_naming_only() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    operation = FindRelationsOperation(tree_clusterer=TreeClusterer(), min_support=0, naming_only=True)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(REL::A<->B (GROUP::A x) (GROUP::B y))'

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    has_simplified = operation.apply(tree)

    assert not has_simplified
    assert str(tree) == '(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))'

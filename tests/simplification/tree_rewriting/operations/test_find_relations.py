from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import FindRelationsOperation
from architxt.tree import Tree


def test_find_relations_simple() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring('(REL::A<->B (GROUP::A x) (GROUP::B y))')


def test_find_relations_collection() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (REL::A<->B (GROUP::A x) (GROUP::B 1)) (REL::A<->B (GROUP::A x) (GROUP::B 2)))'
    )


def test_find_relations_collection_same_group() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))')


def test_find_relations_naming_only() -> None:
    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard, naming_only=True)
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert has_simplified
    assert tree == Tree.fromstring('(REL::A<->B (GROUP::A x) (GROUP::B y))')

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    tree, has_simplified = operation.apply(tree, equiv_subtrees=set())

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

from architxt.similarity import jaccard
from architxt.simplification.tree_rewriting.operations import FindRelationsOperation
from architxt.tree import Tree


def test_find_relations_simple() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    has_simplified = operation.apply(tree, equiv_subtrees={})

    assert has_simplified
    assert str(tree) == '(REL::A<->B (GROUP::A x) (GROUP::B y))'


def test_find_relations_collection() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    has_simplified = operation.apply(tree, equiv_subtrees={})

    assert has_simplified
    assert str(tree) == '(SENT (REL::A<->B (GROUP::A x) (GROUP::B 1)) (REL::A<->B (GROUP::A x) (GROUP::B 2)))'


def test_find_relations_collection_same_group() -> None:
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))')

    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard)
    has_simplified = operation.apply(tree, equiv_subtrees={})

    assert not has_simplified
    assert str(tree) == '(SENT (GROUP::A x) (COLL::A (GROUP::A 1) (GROUP::A 2)))'


def test_find_relations_naming_only() -> None:
    operation = FindRelationsOperation(tau=0.0, min_support=0, metric=jaccard, naming_only=True)
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    has_simplified = operation.apply(tree, equiv_subtrees={})

    assert has_simplified
    assert str(tree) == '(REL::A<->B (GROUP::A x) (GROUP::B y))'

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    has_simplified = operation.apply(tree, equiv_subtrees={})

    assert not has_simplified
    assert str(tree) == '(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))'

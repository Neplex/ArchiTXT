from architxt.operations import find_relationship
from architxt.similarity import jaccard
from architxt.tree import Tree


def test_find_relations_simple():
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    tree, has_simplified = find_relationship(tree, set(), 0.0, 0, jaccard)

    assert not has_simplified
    assert tree == Tree.fromstring('(REL::A<->B (GROUP::A x) (GROUP::B y))')


def test_find_relations_collection():
    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    tree, has_simplified = find_relationship(tree, set(), 0.0, 0, jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (REL::A<->B (GROUP::A x) (GROUP::B 1)) (REL::A<->B (GROUP::A x) (GROUP::B 2)))'
    )


def test_find_relations_naming_only():
    tree = Tree.fromstring('(SENT (GROUP::A x) (GROUP::B y))')

    tree, has_simplified = find_relationship(tree, set(), 0.0, 0, jaccard, naming_only=True)

    assert not has_simplified
    assert tree == Tree.fromstring('(REL::A<->B (GROUP::A x) (GROUP::B y))')

    # =======

    tree = Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

    tree, has_simplified = find_relationship(tree, set(), 0.0, 0, jaccard, naming_only=True)

    assert not has_simplified
    assert tree == Tree.fromstring('(SENT (GROUP::A x) (COLL::B (GROUP::B 1) (GROUP::B 2)))')

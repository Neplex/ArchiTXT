import numpy as np
from architxt.similarity import TreeCluster, TreeClusterer, jaccard
from architxt.simplification.tree_rewriting.operations import FindSubGroupsOperation
from architxt.tree import Tree


def test_find_subgroups_no_simplify() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::X 1) (ENT::Y 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert not has_simplified
    assert str(tree) == '(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))'


def test_find_subgroups_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))'


def test_find_subgroups_simple_group() -> None:
    tree = Tree.fromstring('(SENT (GROUP::1 (ENT::A 1) (ENT::B 2) (ENT::C 3)))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[
                Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),
                Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))'),
            ],
            probabilities=np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    tree[0].label = 'XXX'

    assert has_simplified
    assert str(tree) == '(SENT (XXX (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3)))'


def test_find_subgroups_largest() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4)))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
        '3': TreeCluster(
            trees=[Tree.fromstring('(GROUP::3 (ENT::A 1) (ENT::B 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (ENT::D 4)))'


def test_find_subgroups_multi() -> None:
    tree = Tree.fromstring('(SENT (1 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4) (ENT::E 5)))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
        '3': TreeCluster(
            trees=[Tree.fromstring('(GROUP::3 (ENT::D 4) (ENT::E 5))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))'


def test_find_subgroups_root() -> None:
    tree = Tree.fromstring('(SENT (ENT::A 1) (ENT::B 2) (ENT::C 3))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = FindSubGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3))'

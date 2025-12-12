import numpy as np
from architxt.similarity import TreeCluster, TreeClusterer, jaccard
from architxt.simplification.tree_rewriting.operations import MergeGroupsOperation
from architxt.tree import Tree


def test_merge_groups_simple() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (GROUP::3 (ENT::C 3) (ENT::D 4))))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = MergeGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2) (ENT::C 3) (ENT::D 4))))'


def test_merge_groups_extend() -> None:
    tree = Tree.fromstring('(SENT (1 (GROUP::2 (ENT::A 1) (ENT::B 2)) (ENT::C 3) (GROUP::3 (ENT::D 4) (ENT::E 5))))')

    clusterer = TreeClusterer(tau=0.8, metric=jaccard)
    clusterer._clusters = {
        '2': TreeCluster(
            trees=[Tree.fromstring('(GROUP::2 (ENT::A 1) (ENT::B 2))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
        '4': TreeCluster(
            trees=[Tree.fromstring('(GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3))')],
            probabilities=np.asarray([1.0], dtype=np.float64),
        ),
    }

    operation = MergeGroupsOperation(tree_clusterer=clusterer, min_support=0)
    has_simplified = operation.apply(tree)

    assert has_simplified
    assert str(tree) == '(SENT (1 (GROUP::4 (ENT::A 1) (ENT::B 2) (ENT::C 3)) (GROUP::3 (ENT::D 4) (ENT::E 5))))'

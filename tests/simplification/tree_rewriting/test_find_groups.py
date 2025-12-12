import numpy as np
from architxt.similarity import TreeCluster, TreeClusterer
from architxt.simplification.tree_rewriting import create_group, find_groups
from architxt.tree import Tree


def test_create_group_with_parent() -> None:
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (ENT::Y yyy)))')

    create_group(tree[0], '0')

    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy)))')


def test_create_group_without_parent() -> None:
    tree = Tree.fromstring('(1 (ENT::X xxx) (ENT::Y yyy))')

    create_group(tree, '0')

    assert tree == Tree.fromstring('(GROUP::0 (ENT::X xxx) (ENT::Y yyy))')


def test_create_group_recursive() -> None:
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (2 (ENT::Y yyy) (3 (ENT::Z zzz)))))')

    create_group(tree[0], '0')

    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy) (ENT::Z zzz)))')


def test_find_groups_no_simplification() -> None:
    clusterer = TreeClusterer()
    has_simplified = find_groups(clusterer, min_support=3)

    assert not has_simplified


def test_find_groups_with_parent() -> None:
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (ENT::Y yyy)))')

    clusterer = TreeClusterer()
    clusterer._clusters = {
        '0': TreeCluster(trees=[tree[0]], probabilities=np.asarray([1.0], dtype=np.float64)),
    }

    has_simplified = find_groups(clusterer, min_support=0)

    assert has_simplified
    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy)))')


def test_find_group_without_parent() -> None:
    tree = Tree.fromstring('(SENT (ENT::X xxx) (ENT::Y yyy))')

    clusterer = TreeClusterer()
    clusterer._clusters = {
        '0': TreeCluster(trees=[tree], probabilities=np.asarray([1.0], dtype=np.float64)),
    }

    has_simplified = find_groups(clusterer, min_support=0)

    assert has_simplified
    assert tree == Tree.fromstring('(GROUP::0 (ENT::X xxx) (ENT::Y yyy))')


def test_find_group_largest() -> None:
    tree = Tree.fromstring('(1 (2 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')

    clusterer = TreeClusterer()
    clusterer._clusters = {
        '0': TreeCluster(trees=[tree], probabilities=np.asarray([1.0], dtype=np.float64)),
        '1': TreeCluster(trees=[tree[0]], probabilities=np.asarray([1.0], dtype=np.float64)),
    }

    has_simplified = find_groups(clusterer, min_support=0)

    assert has_simplified
    assert tree == Tree.fromstring('(1 (GROUP::1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')


def test_find_group_frequent() -> None:
    tree = Tree.fromstring('(1 (2 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')

    clusterer = TreeClusterer()
    clusterer._clusters = {
        '1': TreeCluster(trees=[tree], probabilities=np.asarray([1.0], dtype=np.float64)),
        '0': TreeCluster(
            trees=[tree[0], tree[0], tree[0]], probabilities=np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        ),
    }

    has_simplified = find_groups(clusterer, min_support=0)

    assert has_simplified
    assert tree == Tree.fromstring('(1 (GROUP::0 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')


def test_find_groups_multi() -> None:
    tree = Tree.fromstring(
        '(SENT (1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (2 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )

    clusterer = TreeClusterer()
    clusterer._clusters = {
        '1': TreeCluster(trees=[tree[0]], probabilities=np.asarray([1.0], dtype=np.float64)),
        '0': TreeCluster(trees=[tree[2]], probabilities=np.asarray([1.0], dtype=np.float64)),
    }

    has_simplified = find_groups(clusterer, min_support=0)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (GROUP::1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (GROUP::0 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )

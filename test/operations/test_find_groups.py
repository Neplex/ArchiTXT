from architxt.operations import _create_group, find_groups
from architxt.similarity import jaccard
from architxt.tree import Tree


def test_create_group_with_parent():
    tree = Tree.fromstring('(parent (1 (ENT::X xxx)))')

    _create_group(tree[0], 0)

    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx)))')


def test_create_group_without_parent():
    tree = Tree.fromstring('(1 (ENT::X xxx))')

    _create_group(tree, 0)

    assert tree == Tree.fromstring('(1 (GROUP::0 (ENT::X xxx)))')


def test_find_groups():
    tree = Tree.fromstring(
        '(SENT (1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (2 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )
    equiv_subtrees = {
        (tree[0],),
        (tree[2],),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=1, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (GROUP::1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (GROUP::0 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )

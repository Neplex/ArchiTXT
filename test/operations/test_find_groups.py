from architxt.operations import _create_group, find_groups
from architxt.similarity import jaccard
from architxt.tree import Tree


def test_create_group_with_parent():
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (ENT::Y yyy)))')

    _create_group(tree[0], 0)

    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy)))')


def test_create_group_without_parent():
    tree = Tree.fromstring('(1 (ENT::X xxx) (ENT::Y yyy))')

    _create_group(tree, 0)

    assert tree == Tree.fromstring('(GROUP::0 (ENT::X xxx) (ENT::Y yyy))')


def test_create_group_recursive():
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (2 (ENT::Y yyy) (3 (ENT::Z zzz)))))')

    _create_group(tree[0], 0)

    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy) (ENT::Z zzz)))')


def test_find_groups_no_simplification():
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (ENT::Y yyy)))')

    tree, has_simplified = find_groups(tree, set(), tau=0.7, min_support=3, metric=jaccard)

    assert not has_simplified


def test_find_groups_with_parent():
    tree = Tree.fromstring('(parent (1 (ENT::X xxx) (ENT::Y yyy)))')
    equiv_subtrees = {
        (tree[0],),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=0, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(parent (GROUP::0 (ENT::X xxx) (ENT::Y yyy)))')


def test_find_group_without_parent():
    tree = Tree.fromstring('(SENT (ENT::X xxx) (ENT::Y yyy))')
    equiv_subtrees = {
        (tree,),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=0, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(GROUP::0 (ENT::X xxx) (ENT::Y yyy))')


def test_find_group_largest():
    tree = Tree.fromstring('(1 (2 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')
    equiv_subtrees = {
        (tree,),
        (tree[0],),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=0, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(1 (GROUP::1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')


def test_find_group_frequent():
    tree = Tree.fromstring('(1 (2 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')
    equiv_subtrees = {
        (tree,),
        (tree[0], tree[0], tree[0]),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=0, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring('(1 (GROUP::0 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz))')


def test_find_groups_multi():
    tree = Tree.fromstring(
        '(SENT (1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (2 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )
    equiv_subtrees = {
        (tree[0],),
        (tree[2],),
    }

    tree, has_simplified = find_groups(tree, equiv_subtrees, tau=0.7, min_support=0, metric=jaccard)

    assert has_simplified
    assert tree == Tree.fromstring(
        '(SENT (GROUP::1 (ENT::X xxx) (ENT::Y yyy)) (ENT::Z zzz) (GROUP::0 (ENT::A aaa) (ENT::B bbb) (ENT::C ccc)))'
    )

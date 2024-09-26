from architxt.model import NodeLabel, NodeType
from architxt.operations import find_collections, find_subgroups, merge_groups
from architxt.similarity import jaccard
from architxt.tree import ParentedTree


def test_find_collections_simple():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
        ],
    )

    find_collections(tree, None, 0, 0, None)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                NodeLabel(NodeType.COLL, 'A'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
                ],
            )
        ],
    )


def test_find_collections_multi():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
            ParentedTree(NodeLabel(NodeType.GROUP, 'B'), ['3']),
            ParentedTree(NodeLabel(NodeType.GROUP, 'B'), ['4']),
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['5']),
            ParentedTree(NodeLabel(NodeType.ENT), ['6']),
            ParentedTree(NodeLabel(NodeType.REL, 'C'), ['7']),
            ParentedTree(NodeLabel(NodeType.REL, 'C'), ['8']),
        ],
    )

    find_collections(tree, None, 0, 0, None)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                NodeLabel(NodeType.COLL, 'A'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['5']),
                ],
            ),
            ParentedTree(
                NodeLabel(NodeType.COLL, 'B'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'B'), ['3']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'B'), ['4']),
                ],
            ),
            ParentedTree(NodeLabel(NodeType.ENT), ['6']),
            ParentedTree(
                NodeLabel(NodeType.COLL, 'C'),
                [
                    ParentedTree(NodeLabel(NodeType.REL, 'C'), ['7']),
                    ParentedTree(NodeLabel(NodeType.REL, 'C'), ['8']),
                ],
            ),
        ],
    )


def test_find_collections_merge():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
            ParentedTree(
                NodeLabel(NodeType.COLL, 'A'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['3']),
                ],
            ),
            ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['4']),
            ParentedTree(
                NodeLabel(NodeType.COLL, 'A'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['5']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['6']),
                ],
            ),
        ],
    )

    find_collections(tree, None, 0, 0, None)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                NodeLabel(NodeType.COLL, 'A'),
                [
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['2']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['3']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['4']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['5']),
                    ParentedTree(NodeLabel(NodeType.GROUP, 'A'), ['6']),
                ],
            )
        ],
    )


def test_find_subgroups_simple():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                ],
            )
        ],
    )

    equiv_subtrees = {
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '2'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                ],
            ),
        )
    }

    find_subgroups(tree, equiv_subtrees, 0.7, 0, jaccard)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '2'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                        ],
                    ),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                ],
            )
        ],
    )


def test_find_subgroups_multi():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
                ],
            )
        ],
    )

    equiv_subtrees = {
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '2'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                ],
            ),
        ),
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '3'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
                ],
            ),
        ),
    }

    find_subgroups(tree, equiv_subtrees, 0.7, 0, jaccard)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '2'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                        ],
                    ),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '3'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
                        ],
                    ),
                ],
            )
        ],
    )


def test_find_subgroups_largest():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                ],
            )
        ],
    )

    equiv_subtrees = {
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '2'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                ],
            ),
        ),
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '3'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                ],
            ),
        ),
    }

    find_subgroups(tree, equiv_subtrees, 0.7, 0, jaccard)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '2'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                        ],
                    ),
                    ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                ],
            )
        ],
    )


def test_merge_groups_simple():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '2'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                        ],
                    ),
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '3'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                        ],
                    ),
                ],
            )
        ],
    )

    equiv_subtrees = {
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '4'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                ],
            ),
        )
    }

    merge_groups(tree, equiv_subtrees, 0.7, 0, jaccard)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '4'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                        ],
                    )
                ],
            )
        ],
    )


def test_merge_groups_extend():
    tree = ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '2'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                        ],
                    ),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '3'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
                        ],
                    ),
                ],
            )
        ],
    )

    equiv_subtrees = {
        (
            ParentedTree(
                NodeLabel(NodeType.GROUP, '4'),
                [
                    ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                    ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                ],
            ),
        )
    }

    merge_groups(tree, equiv_subtrees, 0.7, 0, jaccard)

    assert tree == ParentedTree(
        'ROOT',
        [
            ParentedTree(
                '1',
                [
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '4'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
                        ],
                    ),
                    ParentedTree(
                        NodeLabel(NodeType.GROUP, '3'),
                        [
                            ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
                            ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
                        ],
                    ),
                ],
            )
        ],
    )

import pytest

from tal_db.model import NodeLabel, NodeType
from tal_db.similarity import jaccard, METRIC_FUNC
from tal_db.tree import ParentedTree

t1 = ParentedTree('ROOT', [
    ParentedTree(NodeLabel(NodeType.GROUP, '1'), [
        ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
        ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
        ParentedTree(NodeLabel(NodeType.ENT, 'C'), ['3']),
        ParentedTree(NodeLabel(NodeType.ENT, 'D'), ['4']),
    ]),
])[0]

t2 = ParentedTree(NodeLabel(NodeType.GROUP, '2'), [
    ParentedTree(NodeLabel(NodeType.ENT, 'E'), ['5']),
    ParentedTree(NodeLabel(NodeType.ENT, 'F'), ['6']),
])

t3 = ParentedTree('ROOT', [
    ParentedTree(NodeLabel(NodeType.GROUP, '1'), [
        ParentedTree(NodeLabel(NodeType.ENT, 'A'), ['1']),
        ParentedTree(NodeLabel(NodeType.ENT, 'B'), ['2']),
    ]),
])[0]


@pytest.mark.parametrize("metric,x,y,expected", [
    (jaccard, 'ABCD', 'ABCD', 1.0),
    (jaccard, 'AABA', 'AABA', 1.0),
    (jaccard, 'A', 'B', 0.0),
    (jaccard, 'B', 'A', 0.0),
    (jaccard, 'AABC', 'BC', 0.5),
    (jaccard, 'BC', 'AABC', 0.5),
    (jaccard, 'AABC', 'AA', 0.5),
    (jaccard, 'AA', 'AABC', 0.5),
])
def test_dist(x: list[str], y: list[str], expected: float, metric: METRIC_FUNC):
    assert metric(x, y) == expected

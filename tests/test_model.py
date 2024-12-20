import pytest
from architxt.model import NodeLabel, NodeType


@pytest.mark.parametrize(
    ('labels', 'count'),
    [
        ((NodeLabel(NodeType.ENT), NodeLabel(NodeType.GROUP)), 2),
        ((NodeLabel(NodeType.ENT, 'label'), NodeLabel(NodeType.GROUP, 'label')), 2),
        ((NodeLabel(NodeType.ENT, 'label'), NodeLabel(NodeType.ENT, 'label')), 1),
        ((NodeLabel(NodeType.ENT, 'label1'), NodeLabel(NodeType.ENT, 'label2')), 2),
    ],
)
def test_label_hash(labels: list[NodeLabel], count: int):
    assert len(set(labels)) == count

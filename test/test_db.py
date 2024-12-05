import pytest
from architxt.db import Schema
from architxt.tree import Tree


@pytest.mark.parametrize(
    ('tree_str', 'keep_unlabelled', 'valid'),
    [
        ('(SENT word1 word2)', False, True),
        ('(SENT word1 word2)', True, True),
        ('(SENT (X word1 word2))', False, True),
        ('(SENT (X word1 word2))', True, False),
        ('(GROUP::1 (ENT::A AAA) (ENT::B bbb))', False, True),
        ('(GROUP::1 (ENT::A AAA) (ENT::B bbb))', True, True),
    ],
)
def test_schema_validity(tree_str: str, keep_unlabelled: bool, valid: bool):
    tree = Tree.fromstring(tree_str)
    schema = Schema.from_forest([tree], keep_unlabelled=keep_unlabelled)
    assert schema.verify() == valid

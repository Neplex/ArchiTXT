import pytest
from architxt.schema import Group, Schema
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
def test_schema_validity(tree_str: str, keep_unlabelled: bool, valid: bool) -> None:
    tree = Tree.fromstring(tree_str)
    schema = Schema.from_forest([tree], keep_unlabelled=keep_unlabelled)
    assert schema.verify() == valid


def test_extract_valid_trees() -> None:
    tree1 = Tree.fromstring('(SENT (GROUP::1 (ENT::A AAA) (ENT::B bbb)) (GROUP::3 (ENT::D DDD)))')
    tree2 = Tree.fromstring(
        '(SENT (GROUP::1 (ENT::A AAA)) (COLL::1 (REL::1 (GROUP::1 (ENT::A AAA)) (GROUP::2 (ENT::C CCC)))))'
    )
    tree3 = Tree.fromstring('(SENT A B C)')
    forest = [tree1, tree2, tree3]

    schema = Schema.from_description(
        groups={
            Group(name='1', entities={'A'}),
            Group(name='2', entities={'C'}),
        },
        collections=False,
    )

    valid_trees = list(schema.extract_valid_trees(forest))

    assert len(valid_trees) == 2
    assert str(valid_trees[0]) == '(ROOT (GROUP::1 (ENT::A AAA)))'
    assert str(valid_trees[1]) == '(ROOT (GROUP::1 (ENT::A AAA)) (GROUP::1 (ENT::A AAA)) (GROUP::2 (ENT::C CCC)))'

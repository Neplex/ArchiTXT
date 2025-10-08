import pandas as pd
import pandas.testing as pdt
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


@pytest.mark.parametrize(
    ('trees_str', 'result'),
    [
        pytest.param(
            [
                '(S (GROUP::Person (ENT::name Alice) (ENT::age 30)))',
                '(S (GROUP::Person (ENT::name Bob) (ENT::age 25)))',
            ],
            {
                'Person': pd.DataFrame(
                    [
                        {'name': 'Alice', 'age': '30'},
                        {'name': 'Bob', 'age': '25'},
                    ]
                ),
            },
            id='one_group',
        ),
        pytest.param(
            [
                '(S (GROUP::Person (ENT::name Alice) (ENT::age 30)))',
                '(S (GROUP::Person (ENT::name Alice) (ENT::age 30)))',
                '(S (GROUP::Person (ENT::name Bob) (ENT::age 25)))',
            ],
            {
                'Person': pd.DataFrame(
                    [
                        {'name': 'Alice', 'age': '30'},
                        {'name': 'Bob', 'age': '25'},
                    ]
                ),
            },
            id='duplicates',
        ),
        pytest.param(
            [],
            {},
            id='empty_forest',
        ),
        pytest.param(
            [
                '(S (ENT::word hello) (ENT::word world))',
            ],
            {},
            id='no_groups',
        ),
        pytest.param(
            [
                '(S (GROUP::Person (ENT::name Alice) (ENT::age 30)) (GROUP::Address (ENT::city NYC) (ENT::zip 10001)))',
                '(S (GROUP::Person (ENT::name Bob) (ENT::age 25)) (GROUP::Address (ENT::city LA) (ENT::zip 90001)))',
            ],
            {
                'Person': pd.DataFrame(
                    [
                        {'name': 'Alice', 'age': '30'},
                        {'name': 'Bob', 'age': '25'},
                    ]
                ),
                'Address': pd.DataFrame(
                    [
                        {'city': 'NYC', 'zip': '10001'},
                        {'city': 'LA', 'zip': '90001'},
                    ]
                ),
            },
            id='multiple_groups',
        ),
        pytest.param(
            [
                '(S (GROUP::Person (ENT::name Alice) (ENT::age 30) (ENT::city NYC)))',
                '(S (GROUP::Person (ENT::name Bob) (ENT::age 25)))',
            ],
            {
                'Person': pd.DataFrame(
                    [
                        {'name': 'Alice', 'age': '30', 'city': 'NYC'},
                        {'name': 'Bob', 'age': '25'},
                    ]
                ),
            },
            id='missing_entities',
        ),
    ],
)
def test_extract_datasets(trees_str: list[str], result: dict[str, pd.DataFrame]) -> None:
    """Test basic dataset extraction from a forest."""
    forest = [Tree.fromstring(tree) for tree in trees_str]

    schema = Schema.from_forest(forest)
    datasets = schema.extract_datasets(forest)

    assert datasets.keys() == result.keys()

    for group, expected in result.items():
        pdt.assert_frame_equal(datasets.get(group), expected, check_dtype=False, check_like=True)

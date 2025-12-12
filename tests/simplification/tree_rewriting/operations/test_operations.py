import pytest
from architxt.schema import Schema
from architxt.similarity import METRIC_FUNC, TreeClusterer, jaccard, jaro, levenshtein
from architxt.simplification.tree_rewriting.operations import (
    FindCollectionsOperation,
    FindRelationsOperation,
    FindSubGroupsOperation,
    MergeGroupsOperation,
    Operation,
    ReduceBottomOperation,
    ReduceTopOperation,
)
from architxt.tree import Tree
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.test_strategies import tree_st


@pytest.mark.parametrize(
    "operation",
    [
        ReduceTopOperation,
        ReduceBottomOperation,
        FindCollectionsOperation,
        FindRelationsOperation,
        MergeGroupsOperation,
        FindSubGroupsOperation,
    ],
)
@settings(deadline=None)
@given(
    forest=st.lists(tree_st(has_parent=False), min_size=5, max_size=20),
    tau=st.floats(min_value=0.1, max_value=1),
    min_support=st.integers(min_value=1, max_value=20),
    metric=st.sampled_from([jaccard, levenshtein, jaro]),
)
def test_operation_behavior(
    forest: list[Tree], tau: float, min_support: int, metric: METRIC_FUNC, operation: type[Operation]
) -> None:
    """
    Test the behavior of operations.

    1. Operations should construct valid labeled structures.
    2. Simplification flags should correctly indicate whether a simplification occurred.
    """
    original_tree = str(forest[0])

    clusterer = TreeClusterer(tau=tau, metric=metric)
    clusterer.fit(forest)

    simplified = operation(tree_clusterer=clusterer, min_support=min_support).apply(forest[0])

    # Check 1: Valid structure
    schema = Schema.from_forest(forest, keep_unlabelled=False)
    note(f'== Schema ==\n{schema.as_cfg()}\n============')
    assert schema.verify(), "Schema verification failed. The operation produced an invalid structure."

    # Check 2: Correct simplification flag
    if simplified:
        assert str(forest[0]) != original_tree, "Simplification flag was True, but the tree remained unchanged."
    else:
        assert str(forest[0]) == original_tree, "Simplification flag was False, but the tree has been changed."

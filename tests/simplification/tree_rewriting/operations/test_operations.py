from copy import deepcopy

import pytest
from architxt.schema import Schema
from architxt.similarity import METRIC_FUNC, equiv_cluster, jaccard, jaro, levenshtein
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
    tree=tree_st(has_parent=False),
    tau=st.floats(min_value=0.1, max_value=1),
    min_support=st.integers(min_value=1, max_value=20),
    metric=st.sampled_from([jaccard, levenshtein, jaro]),
)
def test_operation_behavior(tree: Tree, tau: float, min_support: int, metric: METRIC_FUNC, operation: type[Operation]):
    """
    Test the behavior of operations:
    1. Operations should construct valid labeled structures.
    2. Simplification flags should correctly indicate whether a simplification occurred.
    """
    clusters = equiv_cluster([tree], tau=tau, metric=metric)
    operation = operation(tau=tau, min_support=min_support, metric=metric)

    new_tree, simplified = operation.apply(deepcopy(tree), equiv_subtrees=clusters)

    # Check 1: Valid structure
    schema = Schema.from_forest([new_tree], keep_unlabelled=False)
    note(f'== Schema ==\n{schema.as_cfg()}\n============')
    assert schema.verify(), "Schema verification failed. The operation produced an invalid structure."

    # Check 2: Correct simplification flag
    if simplified:
        assert new_tree.pformat() != tree.pformat(), "Simplification flag was True, but the tree remained unchanged."
    else:
        assert new_tree.pformat() == tree.pformat(), "Simplification flag was False, but the tree has been changed."

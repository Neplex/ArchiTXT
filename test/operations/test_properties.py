from architxt import operations
from architxt.db import Schema
from architxt.operations import OPERATION
from architxt.similarity import METRIC_FUNC, equiv_cluster, jaccard, jaro, levenshtein
from architxt.tree import Tree
from hypothesis import given
from hypothesis import strategies as st

from test.strategies import tree_st

# Reusable strategies for cleaner test definitions
tree_strategy = tree_st(has_parent=False)
tau_strategy = st.floats(min_value=0.1, max_value=1)
min_support_strategy = st.integers(min_value=1, max_value=20)
metric_strategy = st.sampled_from([jaccard, levenshtein, jaro])
operation_strategy = st.sampled_from(
    [
        operations.find_subgroups,
        operations.merge_groups,
        operations.find_collections,
        operations.find_relations,
        operations.reduce_bottom,
        operations.reduce_top,
    ]
)


def perform_operation(
    tree: Tree, tau: float, min_support: int, metric: METRIC_FUNC, operation: OPERATION
) -> tuple[Tree, bool]:
    """
    Helper function to perform an operation and return results.
    This ensures consistent test setup across multiple tests.
    """
    clusters = equiv_cluster([tree], tau=tau, metric=metric)
    return operation(tree, clusters, tau, min_support, metric)


@given(
    tree=tree_strategy,
    tau=tau_strategy,
    min_support=min_support_strategy,
    metric=metric_strategy,
    operation=operation_strategy,
)
def test_operation_constructs_valid_structure(
    tree: Tree, tau: float, min_support: int, metric: METRIC_FUNC, operation: OPERATION
):
    """
    Any given operation should only construct valid labeled structures.
    """
    new_tree, simplified = perform_operation(tree, tau, min_support, metric, operation)

    schema = Schema.from_forest([new_tree], keep_unlabelled=False)
    assert schema.verify(), "Schema verification failed. The operation produced an invalid structure."


@given(
    tree=tree_strategy,
    tau=tau_strategy,
    min_support=min_support_strategy,
    metric=metric_strategy,
    operation=operation_strategy,
)
def test_operation_returns_simplification_flag_correctly(
    tree: Tree, tau: float, min_support: int, metric: METRIC_FUNC, operation: OPERATION
):
    """
    Any given operation should return a positive simplification flag only if it has made a simplification.
    """
    new_tree, simplified = perform_operation(tree, tau, min_support, metric, operation)

    if simplified:
        assert new_tree != tree, "Simplification flag was True, but the tree remained unchanged."

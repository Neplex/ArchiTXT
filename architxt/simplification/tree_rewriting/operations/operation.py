from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanEvent

from architxt.similarity import DECAY, METRIC_FUNC, TREE_CLUSTER, get_equiv_of

if TYPE_CHECKING:
    from architxt.tree import Tree


class Operation(ABC):
    """
    Abstract base class representing a tree rewriting operation.

    This class encapsulates the definition of operations that can be applied
    to a tree structure using certain equivalence subtrees, a threshold value,
    a minimum support value, and a metric function. It acts as the base class
    for any concrete operation and enforces the structure through abstract
    methods.

    :param tau: Threshold for subtree similarity when clustering.
    :param decay: The similarity decay factor.
    :param min_support: The minimum support value for a structure to be considered frequent.
    :param metric: The metric function to use for computing the similarity between subtrees.
    """

    def __init__(self, *, tau: float, min_support: int, decay: float = DECAY, metric: METRIC_FUNC) -> None:
        self.tau = tau
        self.decay = decay
        self.min_support = min_support
        self.metric = metric

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _log_to_mlflow(self, attributes: dict[str, Any]) -> None:
        """
        Log a custom operation event with specified attributes to the active MLflow span.

        If an active span is available, the function attaches a custom event to the span
        for tracking or monitoring in MLflow.

        :param attributes: Dictionary containing key-value pairs representing event attributes.
        """
        if span := mlflow.get_current_active_span():
            event = SpanEvent(self.__class__.__name__, attributes=attributes)
            span.add_event(event)

    def get_equiv_of(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> str | None:
        return get_equiv_of(tree, equiv_subtrees, tau=self.tau, decay=self.decay, metric=self.metric)

    @abstractmethod
    def apply(self, tree: Tree, *, equiv_subtrees: TREE_CLUSTER) -> bool:
        """
        Apply the rewriting operation on the given tree.

        :param tree: The tree to perform the reduction on.
        :param equiv_subtrees: The cluster of equivalent subtrees in the forest.
        :return: A boolean indicating whether the operation modified the tree (True) or left it unaltered (False).
        """

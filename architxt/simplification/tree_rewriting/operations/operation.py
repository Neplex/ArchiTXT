from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import SpanEvent

if TYPE_CHECKING:
    from architxt.similarity import TreeClusterer
    from architxt.tree import Tree


class Operation(ABC):
    """
    Abstract base class representing a tree rewriting operation.

    This class encapsulates the definition of operations that can be applied
    to a tree structure using certain equivalence subtrees, a threshold value,
    a minimum support value, and a metric function. It acts as the base class
    for any concrete operation and enforces the structure through abstract
    methods.

    :param min_support: The minimum support value for a structure to be considered frequent.
    """

    def __init__(self, *, tree_clusterer: TreeClusterer, min_support: int) -> None:
        self.tree_clusterer = tree_clusterer
        self.min_support = min_support

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

    def get_equiv_of(self, tree: Tree) -> str | None:
        return self.tree_clusterer.get_equiv_of(tree)

    def get_class_support(self, equiv_class_name: str) -> int:
        return len(self.tree_clusterer.clusters.get(equiv_class_name, []))

    @abstractmethod
    def apply(self, tree: Tree) -> bool:
        """
        Apply the rewriting operation on the given tree.

        :param tree: The tree to perform the reduction on.
        :return: A boolean indicating whether the operation modified the tree (True) or left it unaltered (False).
        """

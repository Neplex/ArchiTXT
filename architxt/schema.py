from __future__ import annotations

import dataclasses
import math
import warnings
from collections import Counter, defaultdict
from enum import Enum, auto
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.Errors import CancellationException
from antlr4.error.ErrorStrategy import BailErrorStrategy
from nltk import CFG, Nonterminal, Production
from tqdm.auto import tqdm

from architxt.grammar.metagrammarLexer import metagrammarLexer
from architxt.grammar.metagrammarParser import metagrammarParser
from architxt.similarity import jaccard
from architxt.tree import Forest, NodeLabel, NodeType, Tree, TreeOID, has_type

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

__all__ = ['Group', 'Relation', 'RelationOrientation', 'Schema']

_NODE_TYPE_RANK = {
    NodeType.COLL: 1,
    NodeType.REL: 2,
    NodeType.GROUP: 3,
    NodeType.ENT: 4,
}


@dataclasses.dataclass(slots=True, frozen=True)
class Group:
    name: str
    entities: set[str]

    def __hash__(self) -> int:
        return hash(self.name)


class RelationOrientation(Enum):
    """
    Specifies the direction of a relationship between two groups.

    This enum is used to indicate the source or cardinality orientation of a relationship.
    """

    LEFT = auto()
    """The source of the relationship is the left group."""

    RIGHT = auto()
    """The source of the relationship is the right group."""

    BOTH = auto()
    """The relationship is bidirectional or many-to-many, with no single source."""


@dataclasses.dataclass(slots=True, frozen=True)
class Relation:
    name: str
    left: str
    right: str
    orientation: RelationOrientation = RelationOrientation.BOTH

    def __hash__(self) -> int:
        return hash((self.name, self.left, self.right))


class Schema(CFG):
    _groups: set[Group]
    _relations: set[Relation]

    def __init__(self, productions: Iterable[Production], groups: set[Group], relations: set[Relation]) -> None:
        productions = sorted(productions, key=lambda p: Schema._get_rank(p.lhs()))
        root_production = Production(Nonterminal('ROOT'), sorted(prod.lhs() for prod in productions))

        super().__init__(Nonterminal('ROOT'), [root_production, *productions])
        self._groups = groups
        self._relations = relations

    @staticmethod
    def _get_rank(nt: Nonterminal) -> int:
        if isinstance(nt.symbol(), NodeLabel) and nt.symbol().type in _NODE_TYPE_RANK:
            return _NODE_TYPE_RANK[nt.symbol().type]

        return 0

    @classmethod
    def from_description(
        cls,
        *,
        groups: set[Group] | None = None,
        relations: set[Relation] | None = None,
        collections: bool = True,
    ) -> Schema:
        """
        Create a Schema from a description of groups, relations, and collections.

        :param groups: A dictionary mapping groups names to sets of entities.
        :param relations: A dictionary mapping relation names to tuples of group names.
        :param collections: Whether to generate collection productions.
        :return: A Schema object.
        """
        productions: set[Production] = set()

        if groups:
            for group in groups:
                group_label = NodeLabel(NodeType.GROUP, group.name)
                entity_labels = [Nonterminal(NodeLabel(NodeType.ENT, entity)) for entity in group.entities]
                productions.add(Production(Nonterminal(group_label), sorted(entity_labels)))

        if relations:
            for relation in relations:
                relation_label = NodeLabel(NodeType.REL, relation.name)
                group_labels = [
                    Nonterminal(NodeLabel(NodeType.GROUP, relation.left)),
                    Nonterminal(NodeLabel(NodeType.GROUP, relation.right)),
                ]
                productions.add(Production(Nonterminal(relation_label), group_labels))

        if collections:
            coll_productions = {
                Production(Nonterminal(NodeLabel(NodeType.COLL, prod.lhs().symbol().name)), [prod.lhs()])
                for prod in productions
            }
            productions.update(coll_productions)

        return cls(productions, groups or set(), relations or set())

    @classmethod
    def from_forest(cls, forest: Iterable[Tree], *, keep_unlabelled: bool = True, merge_lhs: bool = True) -> Schema:  # noqa: C901
        """
        Create a Schema from a given forest of trees.

        :param forest: The input forest from which to derive the schema.
        :param keep_unlabelled: Whether to keep uncategorized nodes in the schema.
        :param merge_lhs: Whether to merge nodes in the schema.
        :return: A CFG-based schema representation.
        """
        schema_productions: dict[Nonterminal, set[tuple[Nonterminal, ...]]] = defaultdict(set)
        groups: set[Group] = set()
        relations_examples: dict[str, dict[str, dict[str, tuple[TreeOID, TreeOID]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        relations_is_multi: dict[str, dict[str, bool]] = defaultdict(lambda: defaultdict(lambda: False))

        for tree in tqdm(forest, desc='Extract schema', leave=False):
            for prod in tree.productions():
                if prod.is_lexical() or prod.lhs().symbol() == 'ROOT':
                    continue

                if has_type(prod, NodeType.COLL):
                    schema_productions[prod.lhs()] = {(prod.rhs()[0],)}

                elif has_type(prod, NodeType.REL) and len(prod) == 2:
                    rhs = tuple(sorted(prod.rhs()))
                    schema_productions[prod.lhs()].add(rhs)

                elif has_type(prod, NodeType.GROUP):
                    if merge_lhs:
                        merged_rhs = set(prod.rhs()).union(*schema_productions[prod.lhs()])
                        rhs = tuple(sorted(merged_rhs))
                        schema_productions[prod.lhs()] = {rhs}

                    else:
                        rhs = tuple(sorted(set(prod.rhs())))
                        schema_productions[prod.lhs()].add(rhs)

                    group = Group(
                        name=prod.lhs().symbol().name,
                        entities={ent.symbol().name for entities in schema_productions[prod.lhs()] for ent in entities},
                    )
                    groups.add(group)

                elif keep_unlabelled:
                    rhs = tuple(sorted(set(prod.rhs())))
                    schema_productions[prod.lhs()].add(rhs)

            for subtree in tree.subtrees():
                if not has_type(subtree, NodeType.REL) or len(subtree) != 2:
                    continue

                left_tree, right_tree = subtree
                if not has_type(left_tree, NodeType.GROUP) or not has_type(right_tree, NodeType.GROUP):
                    continue

                left, right = sorted((left_tree.oid, right_tree.oid))
                rel = relations_examples[subtree.label.name]

                for child in subtree:
                    relations_is_multi[subtree.label.name][child.label.name] |= False

                    if not (existing := rel[child.label.name].get(child.oid)):
                        rel[child.label.name][child.oid] = (left, right)

                    elif existing != (left, right):
                        relations_is_multi[subtree.label.name][child.label.name] = True

        del relations_examples

        productions = (Production(lhs, rhs) for lhs, alternatives in schema_productions.items() for rhs in alternatives)
        relations = cls._convert_relations(relations_is_multi)

        return cls(productions, groups, relations)

    @cached_property
    def entities(self) -> set[str]:
        """The set of entities in the schema."""
        return {entity for group in self.groups for entity in group.entities}

    @property
    def groups(self) -> set[Group]:
        """The set of groups in the schema."""
        return self._groups

    @property
    def relations(self) -> set[Relation]:
        """The set of relations in the schema."""
        return self._relations

    @staticmethod
    def _convert_relations(
        relations_flags: dict[str, dict[str, bool]],
    ) -> set[Relation]:
        """
        Convert relation counts into relation objects.

        :param relations_flags: A dict mapping relation-name -> { entity: is_multi_flag, ... }
        :return: A set of relations.
        """
        relations: set[Relation] = set()

        for name, flags in relations_flags.items():
            keys = tuple(flags.keys())
            if len(keys) != 2:
                continue

            left, right = keys

            if flags[left] == flags[right]:
                orientation = RelationOrientation.BOTH

            elif flags[left]:
                orientation = RelationOrientation.LEFT

            else:
                orientation = RelationOrientation.RIGHT

            relation = Relation(name=name, left=left, right=right, orientation=orientation)
            relations.add(relation)

        return relations

    def verify(self) -> bool:
        """
        Verify the schema against the meta-grammar.

        :returns: True if the schema is valid, False otherwise.
        """
        input_text = self.as_cfg()

        lexer = metagrammarLexer(InputStream(input_text))
        stream = CommonTokenStream(lexer)
        parser = metagrammarParser(stream)
        parser._errHandler = BailErrorStrategy()

        try:
            parser.start()
            return parser.getNumberOfSyntaxErrors() == 0

        except CancellationException:
            warnings.warn("Invalid syntax")

        except Exception as error:
            warnings.warn(f"Verification failed: {error!s}")

        return False

    @property
    def group_overlap(self) -> float:
        """
        Get the group overlap ratio as a combined Jaccard index.

        The group overlap ratio is computed as the mean of all pairwise Jaccard indices for each pair of groups.

        :return: The group overlap ratio as a float value between 0 and 1.
                 A higher value indicates a higher degree of overlap between groups.
        """
        jaccard_indices = [jaccard(group1.entities, group2.entities) for group1, group2 in combinations(self.groups, 2)]

        # Combine scores (average of pairwise indices)
        return sum(jaccard_indices) / len(jaccard_indices) if jaccard_indices else 0.0

    @property
    def group_balance_score(self) -> float:
        r"""
        Get the balance score of attributes across groups.

        The balance metric (B) measures the dispersion of attributes (coefficient of variation),
        indicating if the schema is well-balanced.
        A higher balance metric indicates that attributes are distributed more evenly across groups, while
        a lower balance metric suggests that some groups may be too large (wide) or too small (fragmented).

        .. math::
            B = 1 - \frac{\sigma(A)}{\mu(A)}

        Where:
            - :math:`A`: The set of attributes counts for all groups.
            - :math:`\mu(A)`: The mean number of attributes per group.
            - :math:`\sigma(A)`: The standard deviation of attribute counts across groups.

        :return: Balance metric (B), a measure of attribute dispersion.
           - :math:`B \approx 1`: Attributes are evenly distributed.
           - :math:`B \approx 0`: Significant imbalance; some groups are much larger or smaller than others.
        """
        if not len(self.groups):
            return 1.0

        entities_counts = [len(group.entities) for group in self.groups]

        mean_attributes = sum(entities_counts) / len(entities_counts)

        variance = sum((count - mean_attributes) ** 2 for count in entities_counts) / len(entities_counts)
        std_dev = math.sqrt(variance)

        variation = std_dev / mean_attributes if mean_attributes else 1.0

        return 1 - variation

    def as_cfg(self) -> str:
        """
        Convert the schema to a CFG representation.

        :returns: The schema as a list of production rules, each terminated by a semicolon.
        """
        return '\n'.join(f"{prod};" for prod in self.productions())

    def extract_valid_trees(self, forest: Iterable[Tree]) -> Generator[Tree, None, None]:
        """
        Filter and return a valid instance (according to the schema) of the provided forest.

        It removes any subtrees with labels that do not match valid labels and gets rid of redundant collections.

        :param forest: The input forest to be cleaned.
        :yield: Valid trees according to the schema.
        """
        valid_labels = (
            {NodeLabel(NodeType.ENT, entity) for entity in self.entities}
            | {NodeLabel(NodeType.GROUP, group.name) for group in self.groups}
            | {NodeLabel(NodeType.REL, rel.name) for rel in self.relations}
        )

        for tree in forest:
            tree = tree.copy()

            # Remove invalid subtrees by promoting their children
            for subtree in tree.subtrees(lambda t: t.label not in valid_labels, include_self=False, reverse=True):
                for child in reversed(subtree):
                    if isinstance(child, Tree):
                        subtree.parent.insert(subtree.parent_index, child.detach())
                subtree.detach()

            # Remove direct leafs from root if it has no type
            if tree.label not in valid_labels:
                tree.label = 'ROOT'
                for child in reversed(tree):
                    if isinstance(child, str):
                        tree.remove(child)

            if tree:
                yield tree

    def extract_datasets(self, forest: Forest) -> dict[str, pd.DataFrame]:
        """
        Extract datasets from a forest for each group defined in the schema.

        :param forest: The input forest to extract datasets from.
        :return: A mapping from group names to datasets.
        """
        datasets: defaultdict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
        groups_names = {group.name for group in self.groups}

        for tree in tqdm(forest, desc='Extract groups datasets', leave=False):
            for group in tree.groups():
                if group in groups_names:
                    datasets[group] = pd.concat(
                        [
                            datasets[group],
                            tree.group_instances(group),
                        ],
                        ignore_index=True,
                    ).drop_duplicates()

        return datasets

    def find_collapsible_groups(self) -> set[str]:
        """
        Identify all groups eligible for collapsing into attributed relationships.

        A group M is collapsible if it participates exactly twice in a 1-n relation
        on the 'one' side, i.e. we want to collapse patterns like:

            A --(n-1)--> M <--(1-n)-- B

        Into a direct n-n edge:

            A --[attributed edge]-- B

        :return: A set of groups that can be turned into attributed edges.

        >>> schema = Schema.from_description(relations={
        ...     Relation(name='R1', left='A', right='M', orientation=RelationOrientation.LEFT),
        ...     Relation(name='R2', left='M', right='B', orientation=RelationOrientation.RIGHT),
        ... })
        >>> schema.find_collapsible_groups()
        {'M'}

        >>> schema = Schema.from_description(relations={
        ...     Relation(name='R1', left='M', right='B', orientation=RelationOrientation.RIGHT),
        ...     Relation(name='R2', left='M', right='C', orientation=RelationOrientation.RIGHT),
        ... })
        >>> schema.find_collapsible_groups()
        {'M'}

        >>> schema = Schema.from_description(relations={
        ...     Relation(name='R1', left='A', right='M', orientation=RelationOrientation.BOTH),
        ...     Relation(name='R2', left='M', right='B', orientation=RelationOrientation.RIGHT),
        ... })
        >>> schema.find_collapsible_groups()
        set()

        >>> schema = Schema.from_description(relations={
        ...     Relation(name='R1', left='A', right='M', orientation=RelationOrientation.LEFT),
        ...     Relation(name='R2', left='M', right='B', orientation=RelationOrientation.RIGHT),
        ...     Relation(name='R2', left='M', right='C', orientation=RelationOrientation.RIGHT),
        ... })
        >>> schema.find_collapsible_groups()
        set()
        """
        group_count: Counter[str] = Counter()

        for relation in self.relations:
            if relation.orientation == RelationOrientation.LEFT:
                group_count[relation.left] += 3
                group_count[relation.right] += 1

            elif relation.orientation == RelationOrientation.RIGHT:
                group_count[relation.left] += 1
                group_count[relation.right] += 3

            else:
                group_count[relation.left] += 3
                group_count[relation.right] += 3

        return {group for group, count in group_count.items() if count == 2}

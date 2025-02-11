import math
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from itertools import combinations

import pandas as pd
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.Errors import CancellationException
from antlr4.error.ErrorStrategy import BailErrorStrategy
from nltk import CFG, Nonterminal, Production

from architxt.grammar.metagrammarLexer import metagrammarLexer
from architxt.grammar.metagrammarParser import metagrammarParser
from architxt.similarity import jaccard
from architxt.tree import Forest, NodeLabel, NodeType, Tree, has_type

__all__ = ['Schema']

NODE_TYPE_RANK = {
    NodeType.COLL: 1,
    NodeType.REL: 2,
    NodeType.GROUP: 3,
    NodeType.ENT: 4,
}


def get_rank(nt: Nonterminal) -> int:
    if isinstance(nt.symbol(), NodeLabel) and nt.symbol().type in NODE_TYPE_RANK:
        return NODE_TYPE_RANK[nt.symbol().type]

    return 0


class Schema(CFG):
    @classmethod
    def from_description(
        cls,
        *,
        groups: dict[str, set[str]] | None = None,
        rels: dict[str, tuple[str, str]] | None = None,
        collections: bool = True,
    ) -> 'Schema':
        """
        Creates a Schema from a description of groups, relations, and collections.

        :param groups: A dictionary mapping groups names to sets of entities.
        :param rels: A dictionary mapping relation names to tuples of group names.
        :param collections: Whether to generate collection productions.
        :return: A Schema object.
        """
        productions = set()

        if groups:
            for group_name, entities in groups.items():
                group_label = NodeLabel(NodeType.GROUP, group_name)
                entity_labels = [Nonterminal(NodeLabel(NodeType.ENT, entity)) for entity in entities]
                productions.add(Production(Nonterminal(group_label), sorted(entity_labels)))

        if rels:
            for relation_name, groups in rels.items():
                relation_label = NodeLabel(NodeType.REL, relation_name)
                group_labels = [Nonterminal(NodeLabel(NodeType.GROUP, group)) for group in groups]
                productions.add(Production(Nonterminal(relation_label), sorted(group_labels)))

        if collections:
            coll_productions = {
                Production(Nonterminal(NodeLabel(NodeType.COLL, prod.lhs().symbol().name)), [prod.lhs()])
                for prod in productions
            }
            productions.update(coll_productions)

        root_prod = Production(Nonterminal('ROOT'), sorted(prod.lhs() for prod in productions))

        return cls(Nonterminal('ROOT'), [root_prod, *sorted(productions, key=lambda p: get_rank(p.lhs()))])

    @classmethod
    def from_forest(cls, forest: Forest, *, keep_unlabelled: bool = True) -> 'Schema':
        """
        Creates a Schema from a given forest of trees.

        :param forest: The input forest from which to derive the schema.
        :param keep_unlabelled: Whether to keep uncategorized nodes in the schema.
        :return: A CFG-based schema representation.
        """
        schema: dict[Nonterminal, set[Nonterminal]] = defaultdict(set)

        for tree in forest:
            for prod in tree.productions():
                # Skip instance and uncategorized nodes
                if prod.is_lexical() or (not keep_unlabelled and not has_type(prod)):
                    continue

                if has_type(prod, NodeType.COLL):
                    schema[prod.lhs()] = {prod.rhs()[0]}

                else:
                    schema[prod.lhs()] |= set(prod.rhs())

        # Create productions for the schema
        productions = [Production(lhs, sorted(rhs)) for lhs, rhs in schema.items()]
        productions = sorted(productions, key=lambda p: get_rank(p.lhs()))

        return cls(Nonterminal('ROOT'), [Production(Nonterminal('ROOT'), sorted(schema.keys())), *productions])

    @cached_property
    def entities(self) -> set[NodeLabel]:
        """The set of entities in the schema."""
        return {
            rhs.symbol() for production in self.productions() for rhs in production.rhs() if has_type(rhs, NodeType.ENT)
        }

    @cached_property
    def groups(self) -> dict[NodeLabel, set[NodeLabel]]:
        """The set of groups in the schema."""
        return {
            production.lhs().symbol(): {entity.symbol() for entity in production.rhs()}
            for production in self.productions()
            if has_type(production, NodeType.GROUP)
        }

    @cached_property
    def relations(self) -> dict[NodeLabel, tuple[NodeLabel, NodeLabel]]:
        """The set of relations in the schema."""
        return {
            production.lhs().symbol(): (production.rhs()[0].symbol(), production.rhs()[1].symbol())
            for production in self.productions()
            if has_type(production, NodeType.REL)
        }

    def verify(self) -> bool:
        """
        Verifies the schema against the meta-grammar.
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
        jaccard_indices = [jaccard(group1, group2) for group1, group2 in combinations(self.groups.values(), 2)]

        # Combine scores (average of pairwise indices)
        return sum(jaccard_indices) / len(jaccard_indices) if jaccard_indices else 0.0

    @property
    def group_balance_score(self) -> float:
        """
        Get the balance score of attributes across groups.

        The balance metric (B) measures the dispersion of attributes (coefficient of variation),
        indicating if the schema is well-balanced.
        A higher balance metric indicates that attributes are distributed more evenly across groups, while
        a lower balance metric suggests that some groups may be too large (wide) or too small (fragmented).

        .. math::
            B = 1 - \frac{\sigma(A)}{\mu(A)}

        Where:
            - :math:`A`: The set of attribute counts for all groups.
            - :math:`\mu(A)`: The mean number of attributes per group.
            - :math:`\sigma(A)`: The standard deviation of attribute counts across groups.

        returns: Balance metric (B), a measure of attribute dispersion.
           - :math:`B \approx 1`: Attributes are evenly distributed.
           - :math:`B \approx 0`: Significant imbalance; some groups are much larger or smaller than others.
        """
        if not len(self.groups):
            return 1.0

        attribute_counts = [len(attributes) for attributes in self.groups.values()]

        mean_attributes = sum(attribute_counts) / len(attribute_counts)

        variance = sum((count - mean_attributes) ** 2 for count in attribute_counts) / len(attribute_counts)
        std_dev = math.sqrt(variance)

        variation = std_dev / mean_attributes if mean_attributes else 1.0

        return 1 - variation

    def as_cfg(self) -> str:
        """
        Converts the schema to a CFG representation.
        :returns: The schema as a list of production rules, each terminated by a semicolon.
        """
        return '\n'.join(f"{prod};" for prod in self.productions())

    def as_sql(self) -> str:
        """
        Converts the schema to an SQL representation.
        TODO: Implement this method.
        :returns: The schema as an SQL creation script.
        """
        raise NotImplementedError

    def as_cypher(self) -> str:
        """
        Converts the schema to a Cypher representation.
        It only define indexes and constraints as properties graph database do not have fixed schema.
        TODO: Implement this method.
        :returns: The schema as a Cypher creation script defining constraints and indexes.
        """
        raise NotImplementedError

    def extract_valid_trees(self, forest: Forest) -> Forest:
        """
        Filters and returns a valid instance (according to the schema) of the provided forest by removing any subtrees
        with labels that do not match valid labels.
        This method also gets rid of redundant collections.

        :param forest: The input forest to be cleaned.
        :return: A list of valid trees according to the schema.
        """
        valid_forest = deepcopy(forest)
        valid_labels = self.entities | self.groups.keys() | self.relations.keys()

        for tree in valid_forest:
            for subtree in reversed(list(tree.subtrees(lambda t: t.label() not in valid_labels))):
                if not (parent := subtree.parent()):
                    subtree.set_label('ROOT')
                    continue

                children = [deepcopy(child) for child in subtree if isinstance(child, Tree)]
                parent.remove(subtree, recursive=False)
                parent.extend(children)

        return valid_forest

    def extract_datasets(self, forest: Forest) -> dict[str, pd.DataFrame]:
        """
        Extract datasets from a forest for each group defined in the schema.

        :param forest: The input forest to extract datasets from.
        :return: A mapping from group names to datasets.
        """
        cleaned_forest = self.extract_valid_trees(forest)

        return {
            group: dataset
            for group in self.groups
            if not (dataset := pd.concat([tree.group_instances(group.name) for tree in cleaned_forest])).empty
        }

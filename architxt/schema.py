from functools import cached_property

from antlr4 import CommonTokenStream, InputStream
from nltk import CFG, Nonterminal, Production

from architxt.grammar.metagrammarLexer import metagrammarLexer
from architxt.grammar.metagrammarParser import metagrammarParser
from architxt.model import NodeLabel, NodeType
from architxt.tree import Forest, has_type

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
        schema: dict[Nonterminal, set[Nonterminal]] = {}

        for tree in forest:
            for prod in tree.productions():
                # Skip instance and uncategorized nodes
                if prod.is_lexical() or (not keep_unlabelled and not has_type(prod)):
                    continue

                if has_type(prod, NodeType.COLL):
                    schema[prod.lhs()] = {prod.rhs()[0]}

                else:
                    schema[prod.lhs()] = schema.get(prod.lhs(), set()) | set(prod.rhs())

        # Create productions for the schema
        productions = [Production(lhs, sorted(rhs)) for lhs, rhs in schema.items()]
        productions = sorted(productions, key=lambda p: get_rank(p.lhs()))

        return cls(Nonterminal('ROOT'), [Production(Nonterminal('ROOT'), sorted(schema.keys())), *productions])

    @cached_property
    def entities(self) -> set[NodeLabel]:
        """The set of entities in the schema."""
        return {
            entity.symbol()
            for production in self.productions()
            if has_type(production, NodeType.GROUP)
            for entity in production.rhs()
            if has_type(entity, NodeType.ENT)
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

        try:
            parser.start()
            return True

        except Exception as e:
            print(f"Verification failed: {e}")
            return False

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

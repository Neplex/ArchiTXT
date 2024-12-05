from functools import cached_property

from antlr4 import CommonTokenStream, InputStream
from nltk import CFG, Nonterminal, Production

from architxt.grammar.metagrammarLexer import metagrammarLexer
from architxt.grammar.metagrammarParser import metagrammarParser
from architxt.model import NodeLabel, NodeType
from architxt.tree import Forest, has_type

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
    def groups(self) -> set[str]:
        """The set of groups in the schema."""
        return {
            str(production.lhs())
            for production in self.productions()
            if has_type(production, {NodeType.GROUP, NodeType.REL})
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

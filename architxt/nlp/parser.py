import asyncio
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterable, Iterable
from copy import deepcopy

import requests.exceptions
from aiostream import stream
from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize.util import align_tokens

from architxt.nlp.entity_resolver import EntityResolver
from architxt.nlp.model import AnnotatedSentence, Entity, Relation, TreeEntity, TreeRel
from architxt.tree import NodeLabel, NodeType, Tree, has_type, reduce_all

__all__ = ['Parser']


class Parser:
    def __init__(
        self,
        *,
        corenlp_url: str,
        max_concurrency: int = 16,
    ):
        """
        :param corenlp_url: The URL of the CoreNLP server.
        :param max_concurrency: Maximum concurrent requests to CoreNLP.
        """
        self.corenlp = CoreNLPParser(url=corenlp_url)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.corenlp.session.close()

    async def parse_batch(
        self,
        sentences: Iterable[AnnotatedSentence] | AsyncIterable[AnnotatedSentence],
        language: str,
        resolver: EntityResolver | None = None,
        batch_size: int | None = None,
    ) -> AsyncGenerator[tuple[AnnotatedSentence, Tree]]:
        """
        Parse a batch of annotated sentences into enriched syntax trees.

        This function processes an iterable (or asynchronous iterable) of sentences, parses each sentence into a
        syntax tree, enriches the tree by resolving coordination structures,
        and applies further enhancements like entity and relation enrichment.
        Optionally, an external entity resolver can be used to unify entities and relations across sentences.

        :param sentences: An iterable or asynchronous iterable of `AnnotatedSentence` objects to be parsed.
        :param language: The language to use for parsing.
        :param resolver: An optional entity resolver used to resolve entities within the parsed trees.
            If `None`, no entity resolution is performed.
        :param batch_size: The maximum number of concurrent parsing tasks that can run at once.
            If `None`, it defaults to the class maximum concurrency parameter.
            It should generally be less than or equal to the maximum concurrency limit.
            It will only load at most `batch_size` element from the input iterable.

        :yields: A tuple of the original `AnnotatedSentence` and its enriched `Tree`.
            Each sentence is parsed independently, and results are yielded as they become available.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    async for sentence, tree in parser.parse_batch(sentences, language="English"):
                        print(sentence)
                        print(tree)
        """
        # Convert sync iterable to async for compatibility
        sentences = stream.iterate(sentences)

        async def task(sent):
            return sent, await self.parse(sent, language=language, resolver=resolver)

        async for sentence, tree in stream.map(
            sentences, task, ordered=False, task_limit=batch_size or self.max_concurrency
        ):
            if tree:
                yield sentence, tree

    async def parse(
        self,
        sentence: AnnotatedSentence,
        *,
        language: str,
        resolver: EntityResolver | None = None,
    ) -> Tree | None:
        """
        Parse an annotated sentence into an enriched syntax tree.

        This function takes an annotated sentence, parses it into a syntax tree, enriches the tree by
        fixing coordination structures, adding extra information (entities and relations), and applying reductions.
        An external entity resolver could be used to unify entities and relations.

        :param sentence: The annotated sentence to parse.
        :param language: The language to use for parsing.
        :param resolver: An optional entity resolver used to resolve entities within the parsed trees.
            If `None`, no entity resolution is performed.

        :returns: An enriched tree object.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    enriched_tree = parse_sentence(sentence, language='English')

        """
        tree = await self.raw_parse(sentence.txt, language)

        if not tree:
            return None

        # Replace specific parenthesis tokens ('-LRB-' for '(', '-RRB-' for ')') in the leaf nodes
        for subtree in tree.subtrees(lambda x: x.height() == 2 and len(x) == 1 and x[0] in {'-LRB-', '-RRB-'}):
            subtree[0] = '(' if subtree[0] == '-LRB-' else ')'

        # Flatten the coordination in the tree structure
        fix_all_coord(tree)

        # Enrich the tree with named entities and relations from the sentence
        try:
            enrich_tree(tree, sentence.txt, sentence.entities, sentence.rels)
        except ValueError as error:
            # Alignment issue, skip the tree
            warnings.warn(f'Alignment issue: {error}')
            return None

        # Reduce the tree structure removing unneeded nodes
        reduce_all(tree, set(NodeType))

        # Don't yield an empty tree
        if not len(tree) or any(isinstance(child, str) for child in tree):
            return None

        assert tree.root().label() == 'SENT'
        assert all(child.label() != 'SENT' for child in tree)

        # Rename nodes to unique undefined names
        # This is needed when measuring statistics
        for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT)):
            subtree.set_label(f'UNDEF_{uuid.uuid4().hex}')

        if resolver:
            await resolve_tree(tree, resolver)

        return tree

    async def raw_parse(self, sentence: str, language: str) -> Tree | None:
        """
        Parses a sentences into syntax trees using CoreNLP server.

        :param sentence: A sentence to parse.
        :param language: The language to use for parsing.

        :returns: The parse tree of the sentence.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    tree = parser.raw_parse(sentence, language='English')
        """
        try:
            async with self.semaphore:
                parse_trees = await asyncio.to_thread(
                    self.corenlp.raw_parse,
                    sentence,
                    properties={
                        'tokenize.language': language,
                        'ssplit.eolonly': 'true',
                    },
                )

            # CoreNLP return a list of candidates tree, we only select the first one.
            # A parse tree may contain multiple sentence subtrees we select only one and convert it into a tree.
            for rooted_tree in parse_trees:
                for sent_tree in rooted_tree:
                    return Tree.convert(sent_tree)

        except requests.exceptions.ConnectionError as error:
            print(f'Cannot parse the following text due to {error.strerror} : "{sentence}"')
            return None


def enrich_tree(tree: Tree, sentence: str, entities: list[Entity], relations: list[Relation]) -> None:
    """
    Enriches a syntactic tree (tree) by inserting entities and relationships, and removing unused subtrees.

    The function processes a list of entities and relations, inserting them into the tree, unnesting entities as needed,
    and finally deleting any subtrees that are not part of the enriched structure.

    :param tree: A tree representing the syntactic tree to enrich.
    :param sentence: The original sentence from which the tree is derived.
    :param entities: A list of `Entity` objects to be inserted into the tree.
    :param relations: A list of `Relation` objects representing the relationships between entities (currently not used).

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB likes) (NP (NNS apples) (CCONJ and) (NNS oranges))))")
    >>> e1 = Entity(name="person", start=0, end=5, id="E1")
    >>> e2 = Entity(name="fruit", start=12, end=18, id="E2")
    >>> e3 = Entity(name="fruit", start=23, end=30, id="E3")
    >>> enrich_tree(t, "Alice likes apples and oranges", [e1, e2, e3], [])
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (NP (ENT::fruit apples) (ENT::fruit oranges))))
    >>> t = Tree.fromstring("(S (NP XXX) (NP YYY))")
    >>> e1 = Entity(name="nested1", start=0, end=3, id="E1")
    >>> e2 = Entity(name="nested2", start=4, end=7, id="E2")
    >>> e3 = Entity(name="overlap", start=0, end=7, id="E3")
    >>> enrich_tree(t, "XXX YYY", [e1, e2, e3], [])
    >>> print(t.pformat(margin=255))
    (S (REL (ENT::overlap XXX YYY) (nested (ENT::nested1 XXX) (ENT::nested2 YYY))))
    """
    assert tree is not None
    assert sentence

    tokens = align_tokens(tree.leaves(), sentence)
    entity_tokens = {
        entity.id: tuple(i for i, token in enumerate(tokens) if entity.start <= token[1] and token[0] < entity.end)
        for entity in entities
    }

    # Insert entities into the tree by length (descending) to handle larger entities first
    computed_spans: set[tuple[int, ...]] = set()
    entity_trees: list[Tree] = []
    for entity in sorted(entities, key=lambda entity: len(entity_tokens[entity.id]), reverse=True):
        entity_span = entity_tokens[entity.id]

        # Check for conflicts and skip problematic entities
        if is_conflicting_entity(entity, entity_span, computed_spans, tree):
            continue

        tree_entity = TreeEntity(entity.name, [tree.leaf_treeposition(i) for i in entity_span])
        entity_tree = ins_ent(tree, tree_entity)
        entity_trees.append(entity_tree)
        computed_spans.add(entity_span)

        for et in entity_trees:
            assert et.parent() is not None, str(et)

    # Unnest any nested entities in reverse order (to avoid modifying parent indices during the process)
    for entity_tree in sorted(entity_trees, key=lambda x: x.height()):
        unnest_ent(entity_tree.parent(), entity_tree.parent_index())

    # Currently, the relation part is commented out, but can be enabled when relations are processed.
    # for relation in relations:
    #     tree_rel = TreeRel((), (), relation.name)
    #     ins_rel(tree, tree_rel)
    if relations:
        warnings.warn("Relations are not yet supported and will be skipped.")

    # Remove subtrees that have no specific entity or relation (i.e., generic subtrees)
    for subtree in list(tree.subtrees(lambda x: x.height() == 2 and not has_type(x))):
        subtree.parent().remove(subtree)


def fix_coord(tree: Tree, pos: int) -> bool:
    """
    Fixes the coordination structure in a tree at the specified position `pos`.
    This function modifies the tree to ensure that the conjunctions are structured correctly
    according to the grammar rules of coordination.

    :param tree: The tree in which coordination adjustments will be made.
    :param pos: The index of the subtree within the parent tree that contains the coordination to fix.
    :return: `True` if the coordination was successfully fixed, `False` otherwise.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_coord(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))
    """
    assert tree is not None

    coord = None

    # Identify the coordination subtree
    for child in tree[pos]:
        if (
            isinstance(child, Tree)
            and child.label() == 'COORD'
            and len(child) > 0
            and isinstance(child[0], Tree)
            and child[0].label() == 'CCONJ'
        ):
            coord = child
            break

    if coord is None:
        return False

    coord_index = coord.parent_index()

    # Create the left and right parts of the conjunction
    left = Tree(tree[pos].label(), children=[Tree.convert(child) for child in tree[pos][:coord_index]])
    right = [Tree.convert(child) for child in coord[1:]]  # Get all NPs after the conjunction

    # Create the conjunction tree
    conj = Tree('CONJ', children=[left, *right])  # CONJ should include the left NP and the conjuncts

    # Insert the new structure back into the tree
    # If children remain on the right of the coordination, we keep the existing level
    new_tree = (
        Tree(tree[pos].label(), children=[conj] + [Tree.convert(child) for child in remaining_children])
        if (remaining_children := tree[pos][coord_index + 1 :])
        else conj
    )

    # Replace the old subtree
    tree[pos] = new_tree

    return True


def fix_conj(tree: Tree, pos: int) -> bool:
    """
    Fixes conjunction structures in a tree at the specified position `pos`.
    If the node at `pos` is labeled 'CONJ', the function flattens any nested conjunctions
    by replacing the node with a new tree that combines its children.

    :param tree: The tree in which the conjunction structure will be fixed.
    :param pos: The index of the 'CONJ' node to be processed.

    :return: `True` if the conjunction structure was modified, `False` otherwise.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))")
    >>> fix_conj(t[1], 1)
    False
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (CONJ (NP (NNS oranges)) (NP (NNS bananas))))))")
    >>> fix_conj(t[1], 1)
    True
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert tree is not None

    # Check if the specified position is valid and corresponds to a 'CONJ' node
    if not isinstance(tree[pos], Tree) or tree[pos].label() != 'CONJ':
        return False

    new_children: list[Tree | str] = []
    # Collect children, flattening nested conjunctions
    for child in tree[pos]:
        if isinstance(child, Tree) and child.label() == 'CONJ':
            new_children.extend(child)  # Extend with children of the nested CONJ
        else:
            new_children.append(child)  # Append non-CONJ children

    # If no changes were made, return False
    if len(new_children) <= len(tree[pos]):
        return False

    # Create a new tree for the flattened conjunction
    new_tree = Tree('CONJ', children=[Tree.convert(t) for t in new_children])

    # Replace the original 'CONJ' node with the new tree
    tree[pos] = new_tree

    return True


def fix_all_coord(tree: Tree) -> None:
    """
    Fixes all coordination structures in a tree.

    This function iteratively applies `fix_coord` and `fix_conj` to the tree
    until no further modifications can be made. It ensures that the tree adheres
    to the correct syntactic structure for coordination and conjunctions.

    :param tree: The tree in which coordination structures will be fixed.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_all_coord(t)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))

    >>> t2 = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges) (COORD (CCONJ and) (NP (NNS bananas))))))))")
    >>> fix_all_coord(t2)
    >>> print(t2.pformat(margin=255))
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    assert tree is not None

    # Fix coordination
    coord_fixed = True
    while coord_fixed:
        coord_fixed = False
        for pos in tree.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix coordination
            if fix_coord(tree[pos[:-1]], pos[-1]):
                coord_fixed = True
                break

    # Fix conjunctions
    conj_fixed = True
    while conj_fixed:
        conj_fixed = False
        for pos in tree.treepositions():
            if len(pos) < 1:
                continue

            # Attempt to fix conjunctions
            if fix_conj(tree[pos[:-1]], pos[-1]):
                conj_fixed = True
                break


def ins_ent(tree: Tree, tree_ent: TreeEntity) -> Tree:
    """
    Inserts a tree entity into the appropriate position within a parented tree. The function modifies the tree
    structure to insert an entity at the correct level based on its positions and root position.

    :param tree: A tree representing the syntactic tree.
    :param tree_ent: A `TreeEntity` containing the entity name and its positions in the tree.
    :return: The updated subtree where the entity was inserted.

    Example:
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent1 = TreeEntity(name="person", positions=[(0, 0)])
    >>> tree_ent2 = TreeEntity(name="fruit", positions=[(1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent1)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (NP (NNS apples))))
    >>> ent_tree = ins_ent(t, tree_ent2)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice) (VP (VB like) (ENT::fruit apples)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (NP Alice) (ENT::xxx like apples))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice apples) (VP (VB like)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like apples))
    >>> t_ent = TreeEntity(name="yyy", positions=[(0, 2)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::xxx Alice like (ENT::yyy apples)))

    >>> t = Tree.fromstring("(S x y z)")
    >>> t_ent = TreeEntity(name="XY", positions=[(0,), (1,)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::XY x y) z)
    >>> t_ent = TreeEntity(name="YZ", positions=[(0, 1), (1,)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t.pformat(margin=255))
    (S (ENT::XY x y) (ENT::YZ y z))
    """
    assert tree is not None

    # Determine the insertion point based on the positions of the entity
    anchor_pos = tree_ent.root_pos
    anchor_pos_len = len(anchor_pos)
    child_pos = tree_ent.positions[0]

    if sum(child_pos[anchor_pos_len + 1 :]) > 0:
        # Entity has children; attach to the common parent at the first child index + 1
        entity_index = child_pos[anchor_pos_len] + 1

    elif (
        tree[tree_ent.root_pos].parent() is None
        or child_pos[anchor_pos_len] > 0
        or tree_ent.positions[-1][anchor_pos_len] < (len(tree[tree_ent.root_pos]) - 1)
    ):
        # Attach to common parent at the correct index
        entity_index = child_pos[anchor_pos_len]

    else:
        # Attach to the grandparent at the common parent index
        entity_index = tree_ent.root_pos[-1]
        anchor_pos = tree_ent.root_pos[:-1]

        # Adjust anchor position upwards if necessary
        while len(tree[anchor_pos]) == 1 and tree[anchor_pos].parent():
            entity_index = anchor_pos[-1]
            anchor_pos = anchor_pos[:-1]

    # Collect and delete children from the original positions
    children = []
    for child_position in reversed(tree_ent.positions):
        parent_position = child_position[:-1]

        if not has_type(tree[parent_position], NodeType.ENT):
            # The entity has no conflict
            children.append(tree[child_position])
            tree[parent_position].pop(child_position[-1], recursive=False)

        elif len(parent_position) <= len(anchor_pos) and parent_position == anchor_pos[: len(parent_position)]:
            # The entity is a child of another
            children.append(tree[child_position])
            tree[parent_position].pop(child_position[-1], recursive=False)

        elif any(
            leaf_position not in tree_ent.positions for leaf_position in tree[parent_position].treepositions('leaves')
        ):
            # The entity overlap with another we need to duplicate overlapping leaves
            # Else, the entity is a parent entity, so we include only leaves not present in nested entities
            children.append(tree[child_position])

    # Create a new tree node for the entity and insert it into the tree
    new_tree = Tree(NodeLabel(NodeType.ENT, tree_ent.name), children=reversed(children))
    tree[anchor_pos].insert(entity_index, new_tree)

    # Return the modified subtree where the entity was inserted
    entity_tree = tree[anchor_pos][entity_index]

    # Remove empty subtree left in place
    for subtree in list(tree.subtrees(lambda st: len(st) == 0)):
        if subtree.parent():
            subtree.parent().remove(subtree)

    return entity_tree


def unnest_ent(tree: Tree, pos: int) -> None:
    """
    Un-nests an entity in a tree at the specified position `pos`.
    If the node at `pos` is labeled as an entity (ENT), the function converts
    the nested structure into a flat structure, creating a relationship (REL)
    between the entity and its nested entities.

    :param tree: The tree in which the entity will be un-nested.
    :param pos: The index of the 'ENT' node to be processed.

    Example:
    >>> t = Tree.fromstring('(S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))')
    >>> unnest_ent(t[0], 0)
    >>> print(t.pformat(margin=255))
    (S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))
    >>> unnest_ent(t, 0)
    >>> print(t.pformat(margin=255))
    (S (REL (ENT::person Alice Bob Charlie) (nested (ENT::person Bob) (ENT::person Charlie))))
    """
    assert tree is not None

    # Check if the specified position corresponds to an 'ENT' node
    if not has_type(tree[pos], NodeType.ENT):
        return

    # Create the main entity tree and collect nested entities
    entity_tree = Tree(tree[pos].label(), children=tree[pos].leaves())

    # Collect nested entities and ensure they are only from the children of the current entity
    nested_entities = [deepcopy(child) for child in tree[pos] if has_type(child, NodeType.ENT)]
    nested_tree = Tree('nested', children=nested_entities)

    # Construct a new relationship tree with the entity and its nested entities
    new_tree = Tree(NodeLabel(NodeType.REL), children=[entity_tree, nested_tree]) if nested_entities else entity_tree

    # Replace the original entity node with the new structure
    tree[pos] = new_tree


def ins_rel(tree: Tree, tree_rel: TreeRel) -> None:
    assert tree is not None


def is_conflicting_entity(
    entity: Entity, entity_span: tuple[int, ...], computed_spans: set[tuple[int, ...]], tree: Tree
) -> bool:
    """
    Checks for conflicts with other entities (overlapping or duplicate spans).
    """
    if entity_span in computed_spans:
        warnings.warn(
            f"Entity {entity.name} with tokens {entity_span} ('{' '.join(tree.leaves()[i] for i in entity_span)}') "
            f"conflicts with a previously inserted entity."
        )
        return True

    for span in computed_spans:
        if any(token in span for token in entity_span) and not all(token in span for token in entity_span):
            warnings.warn(
                f"Entity {entity.name} with tokens {entity_span} ('{' '.join(tree.leaves()[i] for i in entity_span)}') "
                f"partially overlaps with a previously inserted entity with tokens {span} ('{' '.join(tree.leaves()[i] for i in span)}')."
                "Overlapping tokens will be duplicated."
            )
            return False

    return False


async def resolve_tree(tree: Tree, resolver: EntityResolver) -> None:
    """
    Resolve entities in a tree using the provided entity resolver.
    """
    ent_trees = list(tree.subtrees(lambda x: has_type(x, NodeType.ENT)))
    ent_texts = await resolver(' '.join(ent_tree.leaves()) for ent_tree in ent_trees)

    for ent_tree, ent_text in zip(ent_trees, ent_texts):
        ent_tree.clear()
        ent_tree.append(ent_text)

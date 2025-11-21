from __future__ import annotations

import abc
import uuid
import warnings
from typing import TYPE_CHECKING

from aiostream import Stream, pipe, stream
from nltk.tokenize.util import align_tokens
from typing_extensions import Self

from architxt.nlp.model import AnnotatedSentence, Entity, Relation, TreeEntity, TreeRel
from architxt.tree import NodeLabel, NodeType, Tree, has_type

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Collection, Iterable, Iterator, Sequence
    from types import TracebackType

    from architxt.tree import _SubTree

__all__ = ['Parser']


class Parser(abc.ABC):
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        pass

    async def parse_batch(
        self,
        sentences: Iterable[AnnotatedSentence] | AsyncIterable[AnnotatedSentence],
        *,
        language: str,
        batch_size: int = 128,
    ) -> AsyncIterator[tuple[AnnotatedSentence, Tree]]:
        """
        Parse a batch of annotated sentences into enriched syntax trees.

        This function processes an iterable (or asynchronous iterable) of sentences, parses each sentence into a
        syntax tree, enriches the tree by resolving coordination structures,
        and applies further enhancements like entity and relation enrichment.
        Optionally, an external entity resolver can be used to unify entities and relations across sentences.

        :param sentences: An iterable or asynchronous iterable of `AnnotatedSentence` objects to be parsed.
        :param language: The language to use for parsing.
        :param batch_size: The maximum number of concurrent parsing tasks that can run at once.
            It will only load at most `batch_size` elements from the input iterable.

        :yields: A tuple of the original `AnnotatedSentence` and its enriched `Tree`.
            Each sentence is parsed independently, and results are yielded as they become available.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    async for sentence, tree in parser.parse_batch(sentences, language="English"):
                        print(sentence)
                        print(tree)

        """

        def parse(batch_sentences: Sequence[AnnotatedSentence]) -> AsyncIterable[tuple[AnnotatedSentence, Tree]]:
            texts = (sentence.txt for sentence in batch_sentences)
            trees = self.raw_parse(texts, language=language, batch_size=len(batch_sentences))
            return stream.iterate(zip(batch_sentences, trees, strict=True))

        async def process(sent_tree: tuple[AnnotatedSentence, Tree]) -> tuple[AnnotatedSentence, Tree | None]:
            sentence, tree = sent_tree
            return sentence, await process_tree(sentence, tree)

        tree_stream: Stream[tuple[AnnotatedSentence, Tree]] = (
            stream.iterate(sentences)
            | pipe.chunks(batch_size)
            | pipe.flatmap(parse)
            | pipe.amap(process, ordered=False, task_limit=batch_size)
            | pipe.filter(lambda x: x[1] is not None)
        )

        async with tree_stream.stream() as streamer:
            async for result in streamer:
                yield result

    async def parse(
        self,
        sentence: AnnotatedSentence,
        *,
        language: str,
    ) -> Tree | None:
        """
        Parse an annotated sentence into an enriched syntax tree.

        This function takes an annotated sentence, parses it into a syntax tree, enriches the tree by
        fixing coordination structures, adding extra information (entities and relations), and applying reductions.
        An external entity resolver could be used to unify entities and relations.

        :param sentence: The annotated sentence to parse.
        :param language: The language to use for parsing.

        :returns: An enriched tree object.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    tree = parse(sentence, language='English')
                    print(tree)

        """
        for tree in self.raw_parse([sentence.txt], language=language, batch_size=1):
            if enriched_tree := await process_tree(sentence, tree):
                return enriched_tree

        return None

    @abc.abstractmethod
    def raw_parse(self, sentences: Iterable[str], *, language: str, batch_size: int = 64) -> Iterator[Tree]:
        """
        Parse a sentences into syntax trees using CoreNLP server.

        :param sentences: The sentences to parse.
        :param language: The language to use for parsing.
        :param batch_size: The maximum number of concurrent parsing tasks that can run at once.

        :returns: The parse trees of the sentences.

        Example:
            .. code-block:: python

                with Parser(corenlp_url="http://localhost:9000") as parser:
                    for tree in parser.raw_parse(sentences, language='English'):
                        print(tree)

        """
        raise NotImplementedError


async def process_tree(
    sentence: AnnotatedSentence,
    tree: Tree,
) -> Tree | None:
    # Replace specific parenthesis tokens ('-LRB-' for '(', '-RRB-' for ')') in the leaf nodes
    for subtree in tree.subtrees(lambda x: x.height == 2 and len(x) == 1 and x[0] in {'-LRB-', '-RRB-'}):
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

    # Reduce the tree structure by removing unneeded nodes
    tree.reduce_all(set(NodeType))

    # Don't yield an empty tree
    if not len(tree) or any(isinstance(child, str) for child in tree):
        return None

    # Do not keep root with only one child
    if len(tree) == 1 and isinstance(tree[0], Tree):
        tree = tree[0].detach()

    # Rename nodes to unique undefined names
    # This is needed when measuring statistics
    for subtree in tree.subtrees(lambda x: not has_type(x, NodeType.ENT)):
        subtree.label = 'ROOT' if subtree.parent is None else f'UNDEF_{uuid.uuid4().hex}'

    return tree


def enrich_tree(tree: Tree, sentence: str, entities: Collection[Entity], relations: Collection[Relation]) -> None:
    """
    Enriches a syntactic tree (tree) by inserting entities and relationships, and removing unused subtrees.

    The function processes a list of entities and relations, inserting them into the tree, unnesting entities as needed,
    and finally deleting any subtrees that are not part of the enriched structure.

    :param tree: A tree representing the syntactic tree to enrich.
    :param sentence: The original sentence from which the tree is derived.
    :param entities: A list of `Entity` objects to be inserted into the tree.
    :param relations: A list of `Relation` objects representing the relationships between entities (currently not used).

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB likes) (NP (NNS apples) (CCONJ and) (NNS oranges))))")
    >>> e1 = Entity(name="person", start=0, end=5, id="E1", value="alice")
    >>> e2 = Entity(name="fruit", start=12, end=18, id="E2", value="apples")
    >>> e3 = Entity(name="fruit", start=23, end=30, id="E3", value="oranges")
    >>> enrich_tree(t, "Alice likes apples and oranges", [e1, e2, e3], [])
    >>> print(t)
    (S (ENT::person Alice) (VP (NP (ENT::fruit apples) (ENT::fruit oranges))))
    >>> t = Tree.fromstring("(S (NP XXX) (NP YYY))")
    >>> e1 = Entity(name="nested1", start=0, end=3, id="E1", value="xxx")
    >>> e2 = Entity(name="nested2", start=4, end=7, id="E2", value="yyy")
    >>> e3 = Entity(name="overlap", start=0, end=7, id="E3", value="zzz")
    >>> enrich_tree(t, "XXX YYY", [e1, e2, e3], [])
    >>> print(t)
    (S (REL (ENT::overlap XXX YYY) (nested (ENT::nested1 XXX) (ENT::nested2 YYY))))
    """
    tokens = align_tokens(tree.leaves(), sentence)
    entity_tokens = {
        entity.id: tuple(i for i, token in enumerate(tokens) if entity.start <= token[1] and token[0] < entity.end)
        for entity in entities
    }

    # Insert entities into the tree by length (descending) to handle larger entities first
    computed_spans: set[tuple[int, ...]] = set()
    entity_trees: list[_SubTree] = []
    for entity in sorted(entities, key=lambda entity: len(entity_tokens[entity.id]), reverse=True):
        entity_span = entity_tokens[entity.id]

        # Check for conflicts and skip problematic entities
        if is_conflicting_entity(entity, entity_span, computed_spans, tree):
            continue

        tree_entity = TreeEntity(
            name=entity.name,
            value=entity.value,
            positions=[tree.leaf_position(i) for i in entity_span],
        )
        entity_tree = ins_ent(tree, tree_entity)
        entity_trees.append(entity_tree)
        computed_spans.add(entity_span)

    # Unnest any nested entities in reverse order (to avoid modifying parent indices during the process)
    for entity_tree in sorted(entity_trees, key=lambda x: x.height):
        unnest_ent(entity_tree.parent, entity_tree.parent_index)

    # Currently, the relation part is commented out, but can be enabled when relations are processed.
    # for relation in relations:
    #     tree_rel = TreeRel((), (), relation.name)
    #     ins_rel(tree, tree_rel)
    if relations:
        warnings.warn("Relations are not yet supported and will be skipped.")

    # Remove subtrees that have no specific entity or relation (i.e., generic subtrees)
    for subtree in list(tree.subtrees(lambda x: x.height == 2 and not has_type(x), include_self=False)):
        subtree.parent.remove(subtree)


def fix_coord(tree: Tree, pos: int) -> bool:
    """
    Fix the coordination structure in a tree at the specified position `pos`.

    This function modifies the tree to ensure that the conjunctions are structured correctly
    according to the grammar rules of coordination.

    :param tree: The tree in which coordination adjustments will be made.
    :param pos: The index of the subtree within the parent tree that contains the coordination to fix.
    :return: `True` if the coordination was successfully fixed, `False` otherwise.

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_coord(t[1], 1)
    True
    >>> print(t)
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))
    """
    subtree = tree[pos]
    coord: _SubTree | None = None

    # Check if the specified position is valid and corresponds to a subtree
    if not isinstance(subtree, Tree):
        return False

    # Identify the coordination subtree
    for child in subtree:
        if (
            isinstance(child, Tree)
            and child.label == 'COORD'
            and len(child) > 0
            and isinstance(child[0], Tree)
            and child[0].label == 'CCONJ'
        ):
            coord = child
            break

    if coord is None:
        return False

    coord_index = coord.parent_index

    # Create the left and right parts of the conjunction
    left = Tree(
        subtree.label,
        children=[child.detach() if isinstance(child, Tree) else child for child in subtree[:coord_index]],
    )
    right = [child.detach() if isinstance(child, Tree) else child for child in coord[1:]]

    # Create the conjunction tree
    conj = Tree('CONJ', children=[left, *right])  # CONJ should include the left NP and the conjuncts

    # Insert the new structure back into the tree
    # If children remain on the right of the coordination, we keep the existing level
    new_tree = (
        Tree(
            subtree.label,
            children=[conj] + [child.detach() if isinstance(child, Tree) else child for child in remaining_children],
        )
        if (remaining_children := subtree[coord_index + 1 :])
        else conj
    )

    # Replace the old subtree
    tree[pos] = new_tree

    return True


def fix_conj(tree: Tree, pos: int) -> bool:
    """
    Fix conjunction structures in a tree at the specified position `pos`.

    If the node at `pos` is labeled 'CONJ', the function flattens any nested conjunctions
    by replacing the node with a new tree that combines its children.

    :param tree: The tree in which the conjunction structure will be fixed.
    :param pos: The index of the 'CONJ' node to be processed.

    :return: `True` if the conjunction structure was modified, `False` otherwise.

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))")
    >>> fix_conj(t[1], 1)
    False
    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (CONJ (NP (NNS oranges)) (NP (NNS bananas))))))")
    >>> fix_conj(t[1], 1)
    True
    >>> print(t)
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    subtree = tree[pos]

    # Check if the specified position is valid and corresponds to a 'CONJ' node
    if not isinstance(subtree, Tree) or subtree.label != 'CONJ':
        return False

    new_children: list[Tree | str] = []
    # Collect children, flattening nested conjunctions
    for child in subtree:
        if isinstance(child, Tree) and child.label == 'CONJ':
            new_children.extend(child)  # Extend with children of the nested CONJ
        else:
            new_children.append(child)  # Append non-CONJ children

    # If no changes were made, return False
    if len(new_children) <= len(subtree):
        return False

    # Create a new tree for the flattened conjunction
    new_tree = Tree('CONJ', children=[child.detach() if isinstance(child, Tree) else child for child in new_children])

    # Replace the original 'CONJ' node with the new tree
    tree[pos] = new_tree

    return True


def fix_all_coord(tree: Tree) -> None:
    """
    Fix all coordination structures in a tree.

    This function iteratively applies `fix_coord` and `fix_conj` to the tree
    until no further modifications can be made. It ensures that the tree adheres
    to the correct syntactic structure for coordination and conjunctions.

    :param tree: The tree in which coordination structures will be fixed.

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges))))))")
    >>> fix_all_coord(t)
    >>> print(t)
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)))))

    >>> t2 = Tree.fromstring("(S (NP Alice) (VP (VB eats) (NP (NNS apples) (COORD (CCONJ and) (NP (NNS oranges) (COORD (CCONJ and) (NP (NNS bananas))))))))")
    >>> fix_all_coord(t2)
    >>> print(t2)
    (S (NP Alice) (VP (VB eats) (CONJ (NP (NNS apples)) (NP (NNS oranges)) (NP (NNS bananas)))))
    """
    # Fix coordination
    coord_fixed = True
    while coord_fixed:
        coord_fixed = False
        for pos in tuple(tree.positions()):
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
        for pos in tuple(tree.positions()):
            if len(pos) < 1:
                continue

            # Attempt to fix conjunctions
            if fix_conj(tree[pos[:-1]], pos[-1]):
                conj_fixed = True
                break


def ins_ent(tree: Tree, tree_ent: TreeEntity) -> _SubTree:
    """
    Insert a tree entity into the appropriate position within a parented tree.

    The function modifies the tree structure to insert an entity at the correct level
    based on its positions and root position.

    :param tree: A tree representing the syntactic tree.
    :param tree_ent: A `TreeEntity` containing the entity name and its positions in the tree.
    :return: The updated subtree where the entity was inserted.

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent1 = TreeEntity(name="person", positions=[(0, 0)])
    >>> tree_ent2 = TreeEntity(name="fruit", positions=[(1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, tree_ent1)
    >>> print(t)
    (S (ENT::person Alice) (VP (VB like) (NP (NNS apples))))
    >>> ent_tree = ins_ent(t, tree_ent2)
    >>> print(t)
    (S (ENT::person Alice) (VP (VB like) (ENT::fruit apples)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (NP Alice) (ENT::xxx like apples))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::xxx Alice apples) (VP (VB like)))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> t_ent = TreeEntity(name="xxx", positions=[(0, 0), (1, 0, 0), (1, 1, 0, 0)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::xxx Alice like apples))
    >>> t_ent = TreeEntity(name="yyy", positions=[(0, 2)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::xxx Alice like (ENT::yyy apples)))

    >>> t = Tree.fromstring("(S x y z)")
    >>> t_ent = TreeEntity(name="XY", positions=[(0,), (1,)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::XY x y) z)
    >>> t_ent = TreeEntity(name="YZ", positions=[(0, 1), (1,)])
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::XY x y) (ENT::YZ y z))

    >>> t = Tree.fromstring("(S (NP Alice) (VP (VB like) (NP (NNS apples))))")
    >>> tree_ent1 = TreeEntity(name="person", positions=[(0, 0)], value='xxx')
    >>> tree_ent2 = TreeEntity(name="fruit", positions=[(1, 1, 0, 0)], value='yyy')
    >>> ent_tree = ins_ent(t, tree_ent1)
    >>> print(t)
    (S (ENT::person Alice) (VP (VB like) (NP (NNS apples))))
    >>> ent_tree = ins_ent(t, tree_ent2)
    >>> print(t)
    (S (ENT::person Alice) (VP (VB like) (ENT::fruit apples)))
    >>> print(t[0].metadata.get('value'))
    xxx
    >>> print(t[1, 1].metadata.get('value'))
    yyy

    >>> t = Tree.fromstring("(S x y z)")
    >>> t_ent = TreeEntity(name="XY", positions=[(0,), (1,)], value='AAA')
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::XY x y) z)
    >>> t_ent = TreeEntity(name="YZ", positions=[(0, 1), (1,)], value='BBB')
    >>> ent_tree = ins_ent(t, t_ent)
    >>> print(t)
    (S (ENT::XY x y) (ENT::YZ y z))
    >>> print(t[0].metadata.get('value'))
    AAA
    >>> print(t[1].metadata.get('value'))
    BBB
    """
    # Determine the insertion point based on the positions of the entity
    anchor_pos = tree_ent.root_pos
    anchor_pos_len = len(anchor_pos)
    child_pos = tree_ent.positions[0]

    if sum(child_pos[anchor_pos_len + 1 :]) > 0:
        # Entity has children; attach to the common parent at the first child index + 1
        entity_index = child_pos[anchor_pos_len] + 1

    elif (
        not tree_ent.root_pos
        or child_pos[anchor_pos_len] > 0
        or tree_ent.positions[-1][anchor_pos_len] < (len(tree[tree_ent.root_pos]) - 1)
    ):
        # Attach to the common parent at the correct index
        entity_index = child_pos[anchor_pos_len]

    else:
        # Attach to the grandparent at the common parent index
        entity_index = tree_ent.root_pos[-1]
        anchor_pos = tree_ent.root_pos[:-1]

        # Adjust the anchor position upwards if necessary
        while anchor_pos and len(tree[anchor_pos]) == 1:
            entity_index = anchor_pos[-1]
            anchor_pos = anchor_pos[:-1]

    # Collect and delete children from the original positions
    children = []
    for child_position in reversed(tree_ent.positions):
        parent_position = child_position[:-1]
        subtree = tree[parent_position]

        if not isinstance(subtree, Tree):
            msg = "The specified child position does not correspond to a subtree."
            raise TypeError(msg)

        if not has_type(subtree, NodeType.ENT):
            # The entity has no conflict
            children.append(tree[child_position])
            subtree.pop(child_position[-1], recursive=False)

        elif len(parent_position) <= len(anchor_pos) and parent_position == anchor_pos[: len(parent_position)]:
            # The entity is a child of another
            children.append(tree[child_position])
            subtree.pop(child_position[-1], recursive=False)

        elif any(leaf_position not in tree_ent.positions for leaf_position in subtree.positions(order='leaves')):
            # The entity overlap with another we need to duplicate overlapping leaves
            # Else, the entity is a parent entity, so we include only leaves not present in nested entities
            children.append(tree[child_position])

    # Create a new tree node for the entity and insert it into the tree
    new_tree = Tree(
        NodeLabel(NodeType.ENT, tree_ent.name),
        children=reversed(children),
        metadata={
            'value': tree_ent.value,
        },
    )
    tree[anchor_pos].insert(entity_index, new_tree)

    # Return the modified subtree where the entity was inserted
    entity_tree = tree[anchor_pos][entity_index]

    # Remove all empty subtree left in place
    for subtree in list(tree.subtrees(lambda st: len(st) == 0, include_self=False)):
        subtree.parent.remove(subtree)

    return entity_tree


def unnest_ent(tree: Tree, pos: int) -> None:
    """
    Un-nest an entity in a tree at the specified position `pos`.

    If the node at `pos` is labeled as an entity (ENT), the function converts
    the nested structure into a flat structure, creating a relationship (REL)
    between the entity and its nested entities.

    :param tree: The tree in which the entity will be un-nested.
    :param pos: The index of the 'ENT' node to be processed.

    >>> t = Tree.fromstring('(S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))')
    >>> unnest_ent(t[0], 0)
    >>> print(t)
    (S (ENT::person Alice (ENT::person Bob) (ENT::person Charlie)))
    >>> unnest_ent(t, 0)
    >>> print(t)
    (S (REL (ENT::person Alice Bob Charlie) (nested (ENT::person Bob) (ENT::person Charlie))))
    """
    subtree = tree[pos]

    # Check if the specified position corresponds to an 'ENT' node
    if not has_type(subtree, NodeType.ENT):
        return

    # Create the main entity tree and collect nested entities
    entity_tree = Tree(subtree.label, children=subtree.leaves())

    # Collect nested entities and ensure they are only from the children of the current entity
    nested_entities = [child.copy() for child in subtree if has_type(child, NodeType.ENT)]
    nested_tree = Tree('nested', children=nested_entities)

    # Construct a new relationship tree with the entity and its nested entities
    new_tree = Tree(NodeLabel(NodeType.REL), children=[entity_tree, nested_tree]) if nested_entities else entity_tree

    # Replace the original entity node with the new structure
    tree[pos] = new_tree


def ins_rel(tree: Tree, _: TreeRel) -> None:
    pass


def is_conflicting_entity(
    entity: Entity, entity_span: tuple[int, ...], computed_spans: set[tuple[int, ...]], tree: Tree
) -> bool:
    """Check for conflicts with other entities (overlapping or duplicate spans)."""
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

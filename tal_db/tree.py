from nltk.tree import Tree as NLTKTree, ParentedTree as NLTKParentedTree
from nltk.grammar import Production
from tqdm import tqdm

from .model import Entity, Relation, TreeEntity, TreeRel, NodeLabel, NodeType

__all__ = [
    'has_type', 'Tree', 'ParentedTree',
    'ins_elem', 'del_elem', 'reduce', 'reduce_all',
    'fix_coord', 'fix_conj', 'fix_all_coord',
    'ins_ent', 'ins_rel', 'ins_ent_list', 'unnest_ent',
]


class Tree(NLTKTree):
    def __hash__(self) -> int:
        return id(self)

    def entities(self) -> list:
        return list(self.subtrees(lambda st: has_type(st, NodeType.ENT)))

    def merge(self, tree: NLTKTree) -> 'Tree':
        return Tree('ROOT', [
            *Tree.convert(self),
            *Tree.convert(tree),
        ])


class ParentedTree(Tree, NLTKParentedTree):

    def depth(self) -> int:
        return len(self.treeposition()) + 1

    def merge(self, tree: NLTKTree) -> 'ParentedTree':
        merged_tree = super().merge(tree)
        return ParentedTree.convert(merged_tree)


def has_type(t: Tree | Production | NodeLabel, types: set[str] | str | None = None) -> bool:
    if types is None:
        types = set(NodeType)

    elif isinstance(types, str):
        types = {types}

    return (
            isinstance(t, NodeLabel) and t.type in types
    ) or (
            isinstance(t, Production) and isinstance(t.lhs().symbol(), NodeLabel) and t.lhs().symbol().type in types
    ) or (
            isinstance(t, Tree) and isinstance(t.label(), NodeLabel) and t.label().type in types
    )


def ins_elem(t: ParentedTree, x: Tree, pos: int) -> None:
    t.insert(pos, ParentedTree.convert(x))


def del_elem(t: ParentedTree, pos: int) -> None:
    if not isinstance(t, Tree) or t.root() == t:
        return

    t.pop(pos)

    if len(t) == 0:
        del_elem(t.parent(), t.parent_index())


def reduce(t: ParentedTree, pos: int, types: set[NodeType] | None = None) -> bool:
    if not isinstance(t, Tree) or not isinstance(t[pos], Tree) or (types and has_type(t[pos], types)) or (
            not types and len(t[pos]) > 1):
        return False

    children = [Tree.convert(child) for child in t[pos]]
    t.pop(pos)

    for i, child in enumerate(children):
        t.insert(pos + i, ParentedTree.convert(child))

    return True


def reduce_all(t: ParentedTree, skip_types: set[NodeType] | None = None):
    if not isinstance(t, Tree):
        return

    reduced = True
    while reduced:
        reduced = False
        for pos in t.treepositions():
            if len(pos) < 1 or isinstance(t[pos], str) or (skip_types and has_type(t[pos], skip_types)):
                continue

            if reduce(t[pos[:-1]], pos[-1]):
                reduced = True
                break


def fix_coord(t: ParentedTree, pos: int) -> bool:
    if not isinstance(t, Tree):
        return False

    coord = None
    for child in t[pos]:
        if isinstance(child, ParentedTree) and child.label() == 'COORD' and isinstance(child[0], Tree) and child[0].label() == 'CCONJ':
            coord = child
            break

    if not coord:
        return False

    coord_index = coord.parent_index()

    left = Tree(t[pos].label(), children=[Tree.convert(child) for child in t[pos][:coord_index]]) if coord_index > 1 else t[pos][0]
    conjuncts = Tree('CONJUNCTS', children=[Tree.convert(conj) for conj in coord[1:]]) if len(coord) > 2 else coord[1]
    conj = Tree('CONJ', children=[left, Tree.convert(coord[0]), conjuncts])

    if len(t[pos][coord_index + 1:]) > 0:
        new_tree = Tree(t[pos].label(), children=[conj] + [Tree.convert(child) for child in t[pos][coord_index + 1:]])
    else:
        new_tree = conj

    # Replace tree
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree))

    return True


def fix_conj(t: ParentedTree, pos: int) -> bool:
    if not isinstance(t, Tree) or not isinstance(t[pos], Tree) or t[pos].label() != 'CONJ':
        return False

    new_children = []
    for child in t[pos]:
        if isinstance(child, Tree) and child.label() == 'CONJ':
            new_children.extend(child)
        else:
            new_children.append(child)

    if len(new_children) <= len(t[pos]):
        return False

    new_tree = Tree('CONJ', children=[Tree.convert(t) for t in new_children])

    # Replace tree
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree))

    return True


def fix_all_coord(t: ParentedTree):
    if not isinstance(t, Tree):
        return

    # Fix coord
    coord_fixed = True
    while coord_fixed:
        coord_fixed = False
        for pos in t.treepositions():
            if len(pos) < 1:
                continue

            if coord_fixed := fix_coord(t[pos[:-1]], pos[-1]):
                break

    # Fix conj
    conj_fixed = True
    while conj_fixed:
        conj_fixed = False
        for pos in t.treepositions():
            if len(pos) < 1:
                continue

            if conj_fixed := fix_conj(t[pos[:-1]], pos[-1]):
                break


def unnest_ent(t: ParentedTree, pos: int) -> None:
    if not isinstance(t, Tree) or not has_type(t[pos], NodeType.ENT):
        return

    ent_tree = Tree(t[pos].label(), children=t[pos].leaves())
    nested_ents = Tree('nested', children=[Tree.convert(ent_child) for ent_child in t[pos] if has_type(ent_child, NodeType.ENT)])
    new_tree = Tree(NodeLabel(NodeType.REL), children=[ent_tree, nested_ents])

    # Replace tree
    t.pop(pos)
    t.insert(pos, ParentedTree.convert(new_tree if nested_ents else ent_tree))


def ins_ent(t: ParentedTree, tree_ent: TreeEntity) -> ParentedTree | None:
    if not isinstance(t, Tree):
        return None

    if sum(tree_ent.positions[0][len(tree_ent.root_pos) + 1:]) > 0:
        # Attach to common parent at first child index + 1 (as nodes remain at the child index)
        anchor_pos = tree_ent.root_pos
        entity_index = tree_ent.positions[0][len(tree_ent.root_pos)] + 1

    elif tree_ent.positions[0][len(tree_ent.root_pos)] > 0:
        anchor_pos = tree_ent.root_pos
        entity_index = tree_ent.positions[0][len(tree_ent.root_pos)]

    else:
        # As we are the common parent, attach to the grandparent at the common parent index (parent is deleted through recursive deletion)
        entity_index = tree_ent.root_pos[-1]
        anchor_pos = tree_ent.root_pos[:-1]

        while len(t[anchor_pos]) == 1:
            entity_index = anchor_pos[-1]
            anchor_pos = anchor_pos[:-1]

    children = []
    for child_position in reversed(tree_ent.positions):
        child = t[child_position]
        children.append(Tree.convert(child))
        del_elem(t[child_position[:-1]], child_position[-1])

    new_tree = Tree(NodeLabel(NodeType.ENT, tree_ent.name), children=reversed(children))
    ins_elem(t[anchor_pos], new_tree, entity_index)

    return t[anchor_pos][entity_index]


def ins_rel(t: ParentedTree, tree_rel: TreeRel) -> None:
    if not isinstance(t, Tree):
        return


def ins_ent_list(t: ParentedTree, sentence: str, entities: list[Entity], relations: list[Relation]) -> None:
    if not isinstance(t, Tree):
        return

    entities = sorted(entities, key=lambda x: -len(x))
    entities_tree = []

    # Insert entities
    for entity in tqdm(entities, desc='insert entity', leave=False):
        tree_entity = TreeEntity(entity.name, [t.leaf_treeposition(i)[:-1] for i in entity.token_index(sentence, t.leaves())])
        entity_tree = ins_ent(t, tree_entity)
        entities_tree.append(entity_tree)

    # Unnest entities
    for entity_tree in tqdm(reversed(entities_tree), desc='unnest entity', leave=False):
        unnest_ent(entity_tree.parent(), entity_tree.parent_index())

    # Insert relations
    for relation in tqdm(relations, desc='insert relation', leave=False):
        tree_rel = None
        ins_rel(t, tree_rel)

    for subtree in tqdm(list(t.subtrees(lambda x: x.height() == 2 and not has_type(x))), desc='remove leaves', leave=False):
        del_elem(subtree.parent(), subtree.parent_index())

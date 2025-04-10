import pytest
from architxt.database.documents import parse_document_tree
from architxt.tree import Tree


@pytest.mark.parametrize(
    ('document_tree', 'expected'),
    [
        pytest.param(Tree('ROOT', []), [], id='empty'),
        pytest.param(
            Tree.fromstring('(name (ENT::A x) (ENT::B y))'),
            [Tree.fromstring('(GROUP::name (ENT::A x) (ENT::B y))')],
            id='group',
        ),
        pytest.param(
            Tree.fromstring('(COLL::A (ENT::A x) (ENT::A y))'),
            [Tree.fromstring('(GROUP::A (ENT::A x))'), Tree.fromstring('(GROUP::A (ENT::A y))')],
            id='coll',
        ),
        pytest.param(
            Tree.fromstring('(name (ENT::A x) (sub (ENT::B y)))'),
            [Tree.fromstring('(ROOT (REL::name<->sub (GROUP::name (ENT::A x)) (GROUP::sub (ENT::B y))))')],
            id='rel',
        ),
    ],
)
def test_parse_document_tree(document_tree: Tree, expected: list[Tree]) -> None:
    forest = list(parse_document_tree(document_tree))

    assert forest == expected

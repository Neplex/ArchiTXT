import random
import uuid
from collections import OrderedDict
from typing import Any

import pytest
from architxt.database.loader import read_database
from architxt.database.loader.sql import get_oid
from architxt.tree import Forest, Tree
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import Column, Connection, ForeignKey, Integer, MetaData, String, Table

from . import get_connection  # noqa: F401


@pytest.fixture(name='test_db')
def create_test_database(connection: Connection) -> any:
    metadata = MetaData()
    product_table = Table(
        'Product', metadata, Column('id', Integer, primary_key=True), Column('name', String), Column('price', Integer)
    )

    consumer_table = Table(
        'Consumer', metadata, Column('id', Integer, primary_key=True), Column('name', String), Column('age', Integer)
    )

    order_table = Table(
        'Order',
        metadata,
        Column('product_id', Integer, ForeignKey('Product.id'), primary_key=True),
        Column('consumer_id', Integer, ForeignKey('Consumer.id'), primary_key=True),
        Column('quantity', Integer),
    )
    metadata.create_all(connection)

    connection.execute(
        product_table.insert(),
        [
            {'name': 'Laptop', 'price': 1000},
            {'name': 'Smartphone', 'price': 500},
        ],
    )

    connection.execute(
        consumer_table.insert(),
        [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25},
            {'name': 'Charles', 'age': 35},
            {'name': 'David', 'age': 40},
        ],
    )
    connection.execute(
        order_table.insert(),
        [
            {'product_id': 1, 'consumer_id': 1, 'quantity': 2},
            {'product_id': 2, 'consumer_id': 2, 'quantity': 1},
            {'product_id': 1, 'consumer_id': 2, 'quantity': 1},
        ],
    )
    connection.commit()

    yield

    metadata.drop_all(bind=connection)


def _get_expected_trees(include_unreferenced: bool) -> Forest:
    trees = [
        Tree.fromstring(
            """(ROOT
            (GROUP::Order (ENT::quantity 2))
            (REL::Order<->Consumer (GROUP::Order (ENT::quantity 2)) (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30)))
            (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30))
            (REL::Order<->Product (GROUP::Order (ENT::quantity 2)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000)))
            (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
            )"""
        ),
        Tree.fromstring(
            """(ROOT
            (GROUP::Order (ENT::quantity 1))
            (REL::Order<->Consumer (GROUP::Order (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25)))
            (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
            (REL::Order<->Product  (GROUP::Order (ENT::quantity 1)) (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500)))
            (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500))
            )"""
        ),
        Tree.fromstring(
            """(ROOT
            (GROUP::Order (ENT::quantity 1))
            (REL::Order<->Consumer (GROUP::Order (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25)))
            (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
            (REL::Order<->Product  (GROUP::Order (ENT::quantity 1)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000)))
            (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
            )"""
        ),
    ]

    if include_unreferenced:
        trees += [
            Tree.fromstring("(ROOT (GROUP::Consumer (ENT::id 3) (ENT::name Charles) (ENT::age 35)))"),
            Tree.fromstring("(ROOT (GROUP::Consumer (ENT::id 4) (ENT::name David) (ENT::age 40)))"),
        ]

    return trees


@pytest.mark.usefixtures('test_db')
@pytest.mark.parametrize("include_unreferenced", [True, False])
def test_read_database(include_unreferenced: bool, connection: Connection) -> None:
    expected_forest = _get_expected_trees(include_unreferenced)
    forest = read_database(connection, search_all_instances=include_unreferenced)

    for tree, expected in zip(forest, expected_forest):
        assert str(tree) == str(expected)


@given(
    namespace=st.uuids(),
    name=st.text(min_size=1, max_size=100),
    data=st.dictionaries(
        keys=st.text(min_size=1),
        values=st.one_of(st.integers(), st.text(), st.floats(allow_nan=False, allow_infinity=False), st.booleans()),
    ),
)
def test_oid(namespace: uuid.UUID, name: str, data: dict[str, Any]) -> None:
    # Create initial OID
    oid = get_oid(namespace, name, data)

    # Property 1 - Deterministic: same inputs produce the same OID
    assert get_oid(namespace, name, data) == oid

    # Property 2 - Order independence: key order shouldn't matter
    items = list(data.items())
    random.shuffle(items)
    shuffled_data = OrderedDict(items)
    assert get_oid(namespace, name, shuffled_data) == oid

    # Property 3 - Namespace isolation: different database namespaces produce different OIDs
    assert get_oid(uuid.uuid4(), name, data) != oid

    # Property 4 - Table name isolation: different table names produce different OIDs
    assert get_oid(namespace, name + '_different', data) != oid

    # Property 5 - Data isolation: different data produces different OIDs
    modified_data = data.copy()
    if data:
        # Modify the first key's value
        first_key = next(iter(data))
        modified_data[first_key] = str(modified_data[first_key]) + '_modified'
    else:
        # Add a key if dict is empty
        modified_data["test_key"] = "test_value"
    assert get_oid(namespace, name, modified_data) != oid

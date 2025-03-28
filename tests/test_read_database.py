import pytest
from architxt.database import read_database
from architxt.tree import Forest, Tree
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def create_test_database() -> any:
    engine = create_engine('sqlite:///:memory:')
    metadata = MetaData()
    product_table = Table('Product', metadata,
                          Column('id', Integer, primary_key=True),
                          Column('name', String),
                          Column('price', Integer)
                          )

    consumer_table = Table('Consumer', metadata,
                           Column('id', Integer, primary_key=True),
                           Column('name', String),
                           Column('age', Integer)
                           )

    order_table = Table('Order', metadata,
                        Column('product_id', Integer, ForeignKey('Product.id'), primary_key=True),
                        Column('consumer_id', Integer, ForeignKey('Consumer.id'), primary_key=True),
                        Column('quantity', Integer)
                        )
    metadata.create_all(engine)

    session = sessionmaker(bind=engine)()

    session.execute(product_table.insert(), [
        {'name': 'Laptop', 'price': 1000},
        {'name': 'Smartphone', 'price': 500},
    ])

    session.execute(consumer_table.insert(), [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'age': 25},
        {'name': 'Charles', 'age': 35},
        {'name': 'David', 'age': 40},
    ])
    session.execute(order_table.insert(), [
        {'product_id': 1, 'consumer_id': 1, 'quantity': 2},
        {'product_id': 2, 'consumer_id': 2, 'quantity': 1},
        {'product_id': 1, 'consumer_id': 2, 'quantity': 1},
    ])
    session.commit()
    return engine


@pytest.fixture
def build_expected_trees(include_unreferenced: bool) -> Forest:
    if include_unreferenced:
        return [
            Tree.fromstring(
                """(ROOT 
                        (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2))
                        (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
                        (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30))
                        (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))) 
                        (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2)) (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30))) 
                )"""
            ),
            Tree.fromstring(
                """(ROOT 
                        (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1))
                        (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500))
                        (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
                        (REL::Order (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500))) 
                        (REL::Order (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))) 
                )"""
            ),
            Tree.fromstring(
                """(ROOT  
                        (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1))
                        (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
                        (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
                        (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))) 
                        (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25)))
                )"""
            ),
            Tree.fromstring(
                "(ROOT (GROUP::Consumer (ENT::id 3) (ENT::name Charles) (ENT::age 35)))"
            ),
            Tree.fromstring(
                "(ROOT (GROUP::Consumer (ENT::id 4) (ENT::name David) (ENT::age 40)))"
            )
        ]
    return [
        Tree.fromstring(
            """(ROOT 
                    (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2))
                    (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
                    (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30))
                    (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))) 
                    (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 1) (ENT::quantity 2)) (GROUP::Consumer (ENT::id 1) (ENT::name Alice) (ENT::age 30))) 
            )"""
        ),
        Tree.fromstring(
            """(ROOT 
                    (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1))
                    (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500))
                    (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
                    (REL::Order (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Product (ENT::id 2) (ENT::name Smartphone) (ENT::price 500))) 
                    (REL::Order (GROUP::Order (ENT::product_id 2) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))) 
            )"""
        ),
        Tree.fromstring(
            """(ROOT  
                    (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1))
                    (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))
                    (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25))
                    (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Product (ENT::id 1) (ENT::name Laptop) (ENT::price 1000))) 
                    (REL::Order (GROUP::Order (ENT::product_id 1) (ENT::consumer_id 2) (ENT::quantity 1)) (GROUP::Consumer (ENT::id 2) (ENT::name Bob) (ENT::age 25)))
            )"""
        )
    ]


@pytest.mark.parametrize("include_unreferenced", [True, False])
def test_read_database(include_unreferenced: bool, create_test_database: any, build_expected_trees: any) -> None:
    engine = create_test_database

    metadata = MetaData()
    metadata.reflect(bind=engine)

    forest_test = build_expected_trees
    forest = list(read_database(engine, search_all_instances=include_unreferenced))

    for expected_tree, actual_tree in zip(forest_test, forest):
        assert expected_tree.pprint() == actual_tree.pprint()

import pytest
from architxt.database import read_table, read_unreferenced_table
from architxt.tree import Forest, Tree
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, create_engine
from sqlalchemy.orm import sessionmaker


def create_test_database(engine: create_engine) -> None:
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
def test_read_database(include_unreferenced: bool) -> None:
    engine = create_engine('sqlite:///:memory:')

    create_test_database(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    order = metadata.tables['Order']

    with engine.begin() as conn:
        forest = list(read_table(order, conn=conn))
        if include_unreferenced:
            for fk in order.foreign_keys:
                if fk.column.table not in forest:
                    forest.extend(read_unreferenced_table(order, fk, conn=conn, visited_links=set()))

        test_forest = build_expected_trees(include_unreferenced)
        assert len(forest) == len(test_forest)
        for i in range(len(forest)):
            assert isinstance(forest[i], Tree)
            assert len(forest[i]) == len(test_forest[i])
            assert forest[i].depth() == test_forest[i].depth()
            assert forest[i].height() == test_forest[i].height()
            assert forest[i].label() == test_forest[i].label()
            assert forest[i].groups() == test_forest[i].groups()
            assert forest[i].entity_labels() == test_forest[i].entity_labels()
            assert set(forest[i].leaves()) == set(test_forest[i].leaves())
            assert forest[i].has_duplicate_entity() == test_forest[i].has_duplicate_entity()

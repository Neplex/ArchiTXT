from architxt.database import read_table, read_unreferenced_table
from architxt.tree import Tree
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey
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

    Session = sessionmaker(bind=engine)
    session = Session()

    session.execute(product_table.insert(), [
        {'name': 'Laptop', 'price': 1000},
        {'name': 'Smartphone', 'price': 500},
        {'name': 'New Product', 'price': 200},
    ])

    session.execute(consumer_table.insert(), [
        {'name': 'Alice', 'age': 30},
        {'name': 'Bob', 'age': 25},
        {'name': 'Charles', 'age': 35},
    ])
    session.execute(order_table.insert(), [
        {'product_id': 1, 'consumer_id': 1, 'quantity': 2},
        {'product_id': 2, 'consumer_id': 2, 'quantity': 1},
        {'product_id': 1, 'consumer_id': 2, 'quantity': 1},
    ])
    session.commit()


def test_read_database() -> None:
    engine = create_engine('sqlite:///:memory:')

    create_test_database(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    order = metadata.tables['Order']

    with engine.begin() as conn:
        forest = read_table(order, conn=conn)
        for tree in forest:
            assert isinstance(tree, Tree)
            assert len(tree.leaves()) > 0
            tree_str = tree.pformat()
            assert "Order" in tree_str
            tree_leaves = tree.leaves()
            assert "Charles" not in tree_leaves
            assert "New Product" not in tree_leaves
            assert "Bob" in tree_leaves or "Alice" in tree_leaves
            assert "Laptop" in tree_leaves or "Smartphone" in tree_leaves

        rest_forest = []
        for foreign_table in order.foreign_keys:
            rest_forest.extend(read_unreferenced_table(order, foreign_table, conn=conn))

        for tree in rest_forest:
            assert isinstance(tree, Tree)
            assert len(tree.leaves()) > 0
            assert "Order" not in tree.pformat()
            assert "Product" in tree.pformat() or "Consumer" in tree.pformat()
            tree_leaves = tree.leaves()
            assert "Charles" in tree_leaves or "New Product" in tree_leaves

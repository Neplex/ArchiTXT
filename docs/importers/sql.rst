Loading relational databases
============================

.. seealso::

    :doc:`../fundamentals`
        Overview of ArchiTXT's internal data representation.

    :doc:`../examples/database_integration`
        Real-world examples of database integration.

ArchiTXT supports direct integration with relational databases such as SQLite, MySQL, and PostgreSQL.
It automatically converts relational schemas into structured tree representations by analyzing table relationships and extracting corresponding data.

.. code-block:: python

    from sqlalchemy import create_engine
    from architxt.database.loader.sql import read_sql

    with create_engine('postgresql://user:password@localhost:5432/mydb').connect() as connection:
        read_sql(connection)

Process Overview
----------------

ArchiTXT transforms a relational database into a hierarchical structure using graph arboricity with a :abbr:`BFS (Breadth-First Search)` strategy.
The process is designed to build compact :py:class:`~architxt.tree.Tree` that cover the schema efficiently and ensure it fit into memory.
Tables with no incoming foreign keys are selected as roots for the BFS.
These are entry points in the schema graph and cannot be reached from other tables.

For each row of each root table, ArchiTXT perform a BFS, recursively traversing foreign keys.
A foreign key is followed only once per BFS to prevents exponential growth of the :py:class:`~architxt.tree.Tree` and infinite cycle.
The :py:class:`~architxt.tree.Tree` is build during the BFS, where row become group tree named according to the table, attributes become entities, and foreign key links become relationships between groups.
Foreign keys attributes are not part of the group tree.
Each group will be assign an :abbr:`OID (Object Identifier)` based on the name of the database, the name of the table and the primary keys values (see :py:func:`~architxt.database.loader.sql.get_oid` for implementation details).

Once all instances of all root table are exhausted, the process continues for other unvisited tables following a BFS from previous root.
This ensures that all data is eventually covered without redundancy.

Sampling
^^^^^^^^

It is possible to specify the maximum number of instance to read per root table to extract a sample of a large database.

.. code-block:: python

    from sqlalchemy import create_engine
    from architxt.database.loader.sql import read_sql

    with create_engine('postgresql://user:password@localhost:5432/mydb').connect() as connection:
        forest = read_sql(connection, sample=100)  # Only 100 trees will be extracted from root tables


Cyclic Relationships
^^^^^^^^^^^^^^^^^^^^

.. margin::

    .. mermaid::
        :align: center

        ---
        config:
          theme: neutral
        ---
        graph LR
        A --> B
        B --> C
        C --> A

During the BFS traversal, cycles like `A → B → C → A` are avoided by marking visited paths.
If no BFS root can be determine for that cycle, an arbitrary table is chosen as the one with the greater cardinality and the maximum number of instance.

.. margin::

    .. mermaid::
        :align: center

        ---
        config:
          theme: neutral
        ---
        graph LR
            Person --> Person

Self-referencing table will yield a relation between two equivalent groups and can be used as BFS root.

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD
        REL --> Person1
        REL --> Person2
        Person1[Group::Person]
        Person2[Group::Person]
        Person1 --> PersonID1
        PersonID1 --> 1
        Person1 --> Name1
        Name1 --> John
        Person2 --> PersonID2
        PersonID2 --> 2
        Person2 --> Name2
        Name2 --> Jane
        PersonID1[ENT::PersonID]
        Name1[ENT::Name]
        PersonID2[ENT::PersonID]
        Name2[ENT::Name]

Join tables
^^^^^^^^^^^

Join table represent a relationship as a group in **ArchiTXT**, let's consider the following database:

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    erDiagram
        ORDER_DETAIL {
        }

        ORDER {
        }

        PRODUCT {
        }

        ORDER_DETAIL ||--|| ORDER : ""
        ORDER_DETAIL ||--|| PRODUCT : ""

It will be converted to the following tree structure

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD
        ROOT --> REL1[REL]
        ROOT --> REL2[REL]
        REL1 --> A1[GROUP_OrderDetail]
        REL1 --> B[GROUP_Order]
        REL2 --> A2[GROUP_OrderDetail]
        REL2 --> C[GROUP_Product]

But if the join table have no attributes other than the foreign key, it will create an empty group.
So, if the table is link only to two tables (indicating a many-to-many relationship), the table is removed and a relationship between the two linked tables is created.
This behavior can be deactivated using the `simplify_association` parameter.

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD
        REL[REL_OrderDetail] --> B[GROUP_Order]
        REL --> C[GROUP_Product]


Example
-------

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    erDiagram
        ORDER_DETAIL {
        }

        ORDER {
        }

        PRODUCT {
        }

        CONSUMER {
        }

        SUPPLIER {
        }

        ORDER_DETAIL ||--|| ORDER : ""
        ORDER_DETAIL ||--|| PRODUCT : ""
        ORDER ||--|| CONSUMER : ""
        PRODUCT ||--|| SUPPLIER : ""

In this example, `ORDER_DETAIL` is selected as a root table.
For each row in `ORDER_DETAIL`, a BFS builds a :py:class:`~architxt.tree.Tree` that includes related rows from `ORDER`, `PRODUCT`, `CONSUMER`, and `SUPPLIER`.

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD
        ROOT --> RELATION1[REL_OrderDetail]
        ROOT --> RELATION2[REL]
        ROOT --> RELATION3[REL]

        RELATION1 --> ORDER1[GROUP_Order]
        RELATION1 --> PRODUCT1[GROUP_Product]

        RELATION2 --> ORDER2[GROUP_Order]
        RELATION2 --> CONSUMER1[GROUP_Consumer]

        RELATION3 --> PRODUCT2[GROUP_Product]
        RELATION3 --> SUPPLIER2[GROUP_Supplier]

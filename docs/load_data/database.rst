Loading Databases
===========================

ArchiTXT is designed to work seamlessly with relational databases such as SQLite, MySQL, or PostgreSQL. It also works with graph database such as Neo4j. The schema of your database is automatically converted into a tree structure based on the relationships and data extracted.

Database Normalization
----------------------

Your database schema should follow standard normalization rules for databases. These rules are crucial for ensuring data integrity and minimizing redundancy.


Operations
----------

ArchiTXT performs the following operations to transform your database into a structured format:

1. Analyze:
^^^^^^^^^^^^^

ArchiTXT analyzes the schema of the database to determine the optimal way to start the process. It identifies the starting table(s) for the relational database, which are tables that have no foreign keys pointing to them. As for the graph database, it identifies the starting node(s).

2. Read:
^^^^^^^^^^^^^
The data are read from the database. For relational databases, it follows the foreign key relationships to fetch the relevant rows from other tables. For graph databases, it follows the relationships to fetch the relevant nodes and edges. Architxt will also create an unique identifier based on the name of the database and the name of the table and the data or primary key of the table depending on the type of database.

3. Transform:
^^^^^^^^^^^^^
The data are transformed into a structured tree format. The transformation process involves creating a tree structure where the tables are represented as a group, and the rows are represented as entities within those groups. The relationships between the tables are represented as relations between the groups.

.. note:: Refer to the section of the documentation that describes the tree structure of ArchiTXT.

Example
--------

1. Relational Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

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


Here is an example schema diagram representing five tables: ORDER_DETAIL, ORDER, PRODUCT, CONSUMER, and SUPPLIER. ArchiTXT will extract the schema, analyze the relationships, and transform the data into a tree structure.

The program will start by finding the starting table(s) ORDER_DETAIL, which have no foreign keys pointing to it.
The program will read the data from the ORDER_DETAIL table and follow the foreign key relationships to fetch the related data from the ORDER and PRODUCT tables. It will also read the data from the ORDER table and follow the foreign key relationship to fetch the related data from the CONSUMER table. Finally, it will read the data from the PRODUCT table and follow the foreign key relationship to fetch the related data from the SUPPLIER table.
The program will then transform the data into a structured tree format, where each table is represented as a group, and the rows are represented as entities within those groups. The relationships between the tables are represented as relations between the groups.


.. mermaid::

    graph TD
        RELATION1[REL]
        RELATION1 --> ORDER_DETAIL1
        RELATION1 --> ORDER1
        ORDER_DETAIL1[ORDER_DETAIL]
        ORDER1[ORDER]

        RELATION3[REL]
        RELATION3 --> ORDER2
        RELATION3 --> CONSUMER1
        ORDER2[ORDER]
        CONSUMER1[CONSUMER]

        RELATION2[REL]
        RELATION2 --> ORDER_DETAIL2
        RELATION2 --> PRODUCT1
        ORDER_DETAIL2[ORDER_DETAIL]
        PRODUCT1[PRODUCT]

        RELATION4[REL]
        RELATION4 --> PRODUCT2
        RELATION4 --> SUPPLIER2
        PRODUCT2[PRODUCT]
        SUPPLIER2[SUPPLIER]

**Explanation of the Diagram**

This graph illustrates the relationships between the tables. The parent node represents a relation between two tables, and the child nodes represent the tables involved in that relationship.

2. Graph Databases (Neo4j)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. mermaid::

    graph TD
        ORDER_DETAIL[ORDER_DETAIL]
        ORDER[ORDER]
        CONSUMER[CONSUMER]
        SUPPLIER[SUPPLIER]
        ORDER_DETAIL --> ORDER
        ORDER_DETAIL --> PRODUCT
        ORDER --> CONSUMER
        PRODUCT --> SUPPLIER

ArchiTXT can also work with graph databases such as Neo4j. The process is similar to that of relational databases.
The program will read the schema of the graph database and identify the starting node(s) for the graph database. It will then read the data from the starting node(s) and follow the relationships to fetch the relevant nodes and edges.
The program will transform the data into a structured tree format, where each node is represented as a group, and the properties of the nodes are represented as entities within those groups. The relationships between the nodes are represented as relations between the groups.


Particularities
---------------

Case 1: Cyclic Relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    graph LR
    A --> B
    B --> C
    C --> A


In this example of a cyclic relationship between tables A, B, and C, ArchiTXT will detect the cycle and handle it accordingly.
It will take parents from the cycle.

**Self-Referencing Tables**

.. mermaid::

    graph LR
        Person --> Person

In the case of a self-referencing table, ArchiTXT will detect the self-referencing relationship and handle it appropriately.
The relation will be stored in a relational node that references itself.

.. mermaid::

    graph TD
        REL --> Person1
        REL --> Person2
        Person1[Group::Person]
        Person2[Group::Person]
        Person1 --> PersonID1
        PersonID1 --> 1
        Person1 --> Name1
        Name1 --> John
        ID[2]
        Person2 --> PersonID2
        PersonID2 --> 2
        Person2 --> Name2
        Name2 --> Jane
        PersonID1[ENT::PersonID]
        Name1[ENT::Name]
        PersonID2[ENT::PersonID]
        Name2[ENT::Name]


Case 2: Multiple Relationships / Many-to-Many (Relational Database)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

    graph TD
        A --- B
        A --- C

When a table has only foreign keys pointing to other tables, you can activate a flag in the command to remove this table.

.. mermaid::

    graph TD
        REL1[REL]
        REL1 --> A1
        REL1 --> B
        REL2[REL]
        REL2 --> A2
        REL2 --> C
        A1[A]
        A2[A]

After the removal of the table A in the example above, the resulting structure will look like this:

.. mermaid::

    graph TD
        REL1[REL]
        REL1 --> B1
        REL1 --> C1
        REL2[REL]
        REL2 --> C2
        REL2 --> B2
        B1[B]
        C1[C]
        B2[B]
        C2[C]

Case 3: Labels for the Relationships (Graph Database)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the case of a graph database, a node can have multiple labels. ArchiTXT will perform only on the first label of the node.

Export as relational databases
==============================

.. seealso::

    :doc:`../fundamentals`
        A description of the internal representation of data in **ArchiTXT**

**ArchiTXT** intermediate model can be exported to any relational database by interpreting the metamodel schema and persisting the data into normalized tables using the :py:func:`~architxt.database.export.sql.export_sql` function.
The exporter reads the a collection of :py:class:`~architxt.tree.Tree` to generate the SQL schema.
It then reads the trees and inserts rows in the correct order, respecting referential integrity.

.. code-block:: python

    from sqlalchemy import create_engine
    from architxt.database.export.sql import export_sql

    with create_engine('postgresql://user:password@localhost:5432/mydb').connect() as connection:
        export_sql(forest, connection)

Process Overview
----------------

#. Extract the metamodel schema and translate it to the relational database schema.
   Many-to-many relations in the metamodel create a join tables with composite foreign keys.
#. Create and execute SQL statements to build the relational schema.
#. Traverse each :py:class:`~architxt.tree.Tree` and insert data into the database.
   One transaction per tree is used.

Schema creation
^^^^^^^^^^^^^^^

The exporter begins by parsing the forest to derive the relational structure:

Groups → Tables
    Each group in the metamodel becomes a table.
    The table name defaults to the group name (e.g., Order, Product).

Entities → Columns
    Entity leaves under a group define columns.
    Data types (string, integer, date) are inferred or explicitly specified in the metamodel.

Primary Keys
    Each table has one primary key default named as `architxt_<group>_id` that store the :py:class:`~architxt.tree.Tree` :abbr:`OID (Object Identifier)`.
    The naming can be customize using a custom `pk_factory` in :py:func:`~architxt.database.export.sql.export_sql`.

One-to-Many Relationships → Foreign Keys
    If a group `A` is link to a group `B` in a 1-n relation, the table for `A` will have a foreign key column `<B>_id` referencing `B(b_id)`.

Many-to-Many Associations → Join Tables
    If two groups `A` and `B`are connected by a many-to-many relationship or via an association group, the exporter creates an intermediate join table.
    This table includes two foreign keys pointing to the related tables.

Data import
^^^^^^^^^^^

Once the schema is established in the database, the exporter iterates over each :py:class:`~architxt.tree.Tree` to collect data instances.
Group entities are processed first, followed by relations whose referenced groups have already been handled.
Data is then inserted in dependency order: starting with records that have no foreign keys, and progressively inserting those with resolved dependencies until all data is persisted.

.. warning::

    The exporter stages rows in memory to resolve table dependencies and determine a valid insertion sequence.
    It would not be able to import :py:class:`~architxt.tree.Tree` larger than the available memory.

Example
-------

Given the following schema with all relations having 1-n cardinality:

.. productionlist::
    REL1: GROUP_Order GROUP_Order_Detail
    REL2: GROUP_Product GROUP_Order_Detail
    REL3: GROUP_Order GROUP_Consumer
    REL4: GROUP_Product GROUP_Supplier
    GROUP_Order: ENT_order_date ENT_status
    GROUP_Product: ENT_name ENT_description ENT_price
    GROUP_Order_Detail: ENT_quantity ENT_price
    GROUP_Consumer: ENT_name ENT_email ENT_address ENT_phone
    GROUP_Supplier: ENT_name ENT_email ENT_address ENT_phone

The exporter will produce the following relational schema:

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    erDiagram
        ORDER_DETAIL {
            uuid architxt_ORDER_DETAIL_id PK
            uuid architxt_ORDER_id FK
            uuid architxt_PRODUCT_id FK
            string quantity
            string price
        }

        ORDER {
            uuid architxt_ORDER_id PK
            uuid architxt_CONSUMER_id FK
            string order_date
            string status
        }

        PRODUCT {
            uuid architxt_PRODUCT_id PK
            uuid architxt_SUPPLIER_id FK
            string name
            string description
            string price
        }

        CONSUMER {
            uuid architxt_CONSUMER_id PK
            string name
            string email
            string address
            string phone
        }

        SUPPLIER {
            uuid architxt_SUPPLIER_id PK
            string name
            string email
            string address
            string phone
        }

        ORDER_DETAIL }|--|| ORDER : ""
        ORDER_DETAIL }|--|| PRODUCT : ""
        ORDER }|--|| CONSUMER : ""
        PRODUCT }|--|| SUPPLIER : ""

Notes
-----

Table many to many 3 foreign key or more
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a many-to-many relationship with more than two foreign keys, a separate table will be created if a group exists.
This table will include an ID column and all the foreign keys.

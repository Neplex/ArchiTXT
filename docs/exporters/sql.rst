Export as relational databases
==============================

To export data to a relational database, the program will read the metamodel and export the data to the relational database. The program will create tables in the relational database according to the schema defined in the metamodel.

Process Overview
----------------

1. **Reading the schema**: The program starts by reading the schema of the metamodel to determine the relational database schema. The schema describes the tables, their columns, and the relationships to create the relational database.
2. **Creating the database schema**: The program creates the database schema in the graph database. The schema will create the tables of the database. Each table in the schema will be created with columns corresponding to the attribute of the entities in the model.
3. **Reading the data**: The program reads the data from the metamodel.
4. **Exporting the data**: The program exports the data to the graph database. The data will be exported as nodes and relationships in the graph database.

1. Reading the schema
---------------------

The program reads the schema of the metamodel to determine the relational database schema. Firstly, the program reads the schema of the metamodel. Secondly, we determine which tables are a many to many relationship. Finally, we create the database schema in the graph database. The schema will create the tables of the database. Each table in the schema will be created with columns corresponding to the entities in the model.

2. Creating the database schema
-------------------------------

The program will create the database schema based on the schema of the metamodel. It will create tables based on the groups defined in the metamodel. It will also add a primary key to each table. The primary key will be the name of the table with the suffix "_id". The program will also create the columns of the tables. The columns will be created based on the attributes of the entities in the model. The program will also add foreign keys to the tables. The foreign keys will be created based on the relationships of the metamodel. The program will create a column for each foreign key in the table. The name of the column will be as default with prefix architxt_ and ID as suffix. Additionally, it will add the foreign keys corresponding to the relationships of the metamodel.

.. note::
    For the many to many relationships, the program will create a new table to represent the relationship. This table will have two foreign keys pointing to the two tables.

3. Reading the data
-------------------

The program reads the data from the metamodel. It will read the data from each tree of the metamodel. We start by reading the groups of the metamodel and add it to the data to export. Then, we read the relation of the metamodel and add the foreign keys to the data to export that have already been read.

.. note::
    The data to export will follow the schema created in the previous step.

4. Exporting the data
---------------------

The program exports the data from the metamodel read before from the tree and export it to the relational database. It will start by inserting the data that have no foreign keys. Then, it will insert the data that have foreign keys. The program will insert the data who have foreign keys that have already been inserted before and will continue until all the data have been inserted.


**Example of the current database for the metamodel that has been exported to a relational database:**


.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    erDiagram
        ORDER_DETAIL {
            string architxt_ORDER_DETAIL_id PK
            string architxt_ORDER_id FK
            string architxt_PRODUCT_id FK
            string quantity
            string price
        }

        ORDER {
            string architxt_ORDER_id PK
            string architxt_CONSUMER_id FK
            string order_date
            string status
        }

        PRODUCT {
            string architxt_PRODUCT_id PK
            string architxt_SUPPLIER_id FK
            string name
            string description
            string price
        }

        CONSUMER {
            string architxt_CONSUMER_id PK
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

Particularities
---------------

Table many to many 3 foreign key or more
    For a many-to-many relationship with more than two foreign keys, a separate table will be created if a group exists. This table will include an ID column and all the foreign keys.

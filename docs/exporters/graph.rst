Export as property graph
========================

The **ArchiTXT** intermediate model can be exported as a property graph using the :py:func:`architxt.database.export.graph.export_graph` function.
It converts a collection of :py:class:`~architxt.tree.Tree` objects into a property graph, following the structure described in :doc:`importer/graph`.

.. note:: The exporter is compatible with any Neo4J-compliant database management system that supports the Cypher query language and the Bolt protocol.

.. code-block:: python

    from neo4j import GraphDatabase
    from architxt.database.export.graph import export_graph

    with (
        GraphDatabase.driver('bolt://localhost:7687') as driver,
        driver.session() as session,
    ):
        export_graph(forest, session=session)

Process Overview
----------------

To export data to a graph database, the exporter traverses all the trees one by one and persists them as a set of nodes and relationships:

Nodes
    For each group tree, a corresponding node is created, with it's children as properties.
    Nodes are created only once per OID and properties are merged together.

Relationship
    Each relation tree is translated into a graph relationship linking the appropriate nodes.

Example
-------

Consider the following :py:class:`~architxt.tree.Tree`:

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD
        ROOT[ROOT]
        ROOT --> RELATION1[REL1]
        ROOT --> RELATION2[REL2]

        RELATION1 --> ORDER_DETAIL1[GROUP OrderDetail]
        ORDER_DETAIL1 --> OD1QT[ENT quantity] --> OD1QT_VAL[12]
        ORDER_DETAIL1 --> OD1P[ENT price] --> OD1P_VAL[14]

        RELATION1 --> ORDER1[Group Order]
        ORDER1 --> DATE[ENT orderDate] --> DATE_VAL[2016-07-04]
        ORDER1 --> COUNTRY[ENT shipCountry] --> COUNTRY_VAL[France]

        RELATION2 --> ORDER_DETAIL2[GROUP OrderDetail]
        ORDER_DETAIL2 --> OD2QT[ENT quantity] --> OD2QT_VAL[12]
        ORDER_DETAIL2 --> OD2P[ENT price] --> OD2P_VAL[14]

        RELATION2 --> PRODUCT1[Group Product]
        PRODUCT1 --> NAME[ENT name] --> NAME_VAL[Queso Cabrales]
        PRODUCT1 --> STOCK[ENT stock] --> STOCK_VAL[22]

The exporter will build the following property graph :

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    %%{ init: { "securityLevel": "loose", "flowchart": { "htmlLabels": true } } }%%
    flowchart LR
        Order(("<b>Order</b><br/>orderDate: 2016-07-04<br/>shipCountry: France"))
        Product(("<b>Product</b><br/>name: Queso Cabrales<br/>stock: 22"))
        Order -->|"<b>OrderDetail</b><br/>quantity: 12<br/>price: 14"| Product

Export as property graph
========================

.. seealso::

    :doc:`../fundamentals`
        A description of the internal representation of data in **ArchiTXT**

The **ArchiTXT** intermediate model can be exported as a property graph using the :py:func:`~architxt.database.export.cypher.export_cypher` function.
It converts a collection of :py:class:`~architxt.tree.Tree` objects into a property graph, compatible with any Neo4j-compliant database that supports Cypher and the Bolt protocol.

.. code-block:: python

    from neo4j import GraphDatabase
    from architxt.database.export.cypher import export_cypher

    with (
        GraphDatabase.driver('bolt://localhost:7687') as driver,
        driver.session() as session,
    ):
        export_cypher(forest, session=session)

Process Overview
----------------

The exporter processes the forest of trees as follows:

#. Group nodes are converted into graph nodes using :abbr:`OIDs (Object Identifier)` to ensure uniqueness.
#. Relationship Mapping: Relation nodes are translated into directed edges between group nodes, inheriting labels and properties.
#. Entity are interpreted as key-value properties for nodes or relationships

Nodes and Properties
^^^^^^^^^^^^^^^^^^^^

- Nodes are labeled based on the group name (e.g., Order, Product).
- Unique nodes are created per OID; additional properties are merged.
- Entity become properties of their parent node.

Relationships
^^^^^^^^^^^^^

- Label is inherited from the relation name (e.g., OrderDetail).
- Multiple identical relationships (same OID pair and label) are merged.

Due to the semantics of the ArchiTXT model, where a relationship is strictly between two groups and cannot directly carry its own properties, intermediate nodes (analogous to association or join tables in a relational schema) are used to store attributes.
When exporting to a property graph, we can represent these link nodes as attributed edges, which provide a richer and more concise relationship model.
An intermediate group is eligible for collapsing if it participates in exactly two 1-n relations on the "one" side—forming a pattern such as:

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    classDiagram
        class A
        class B
        class M

        A "1..n" --> "1" M : R1
        B "1..n" --> "1" M : R2

This structure can be collapsed into:

.. mermaid::
    :align: center

    ---
    config:
      theme: neutral
    ---
    classDiagram
        class A
        class B

        A "1..n" --> "1..n" B : M

This transformation simplifies the resulting property graph by encoding metadata directly on the relationship rather than as a separate node.
The method :py:meth:`~architxt.schema.Schema.find_collapsible_groups` identifies these link-groups that can be transformed into attributed edges.

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

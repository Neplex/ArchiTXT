Loading property graphs
=======================

.. seealso::

    :doc:`../fundamentals`
        Overview of ArchiTXT's internal data representation.

ArchiTXT supports loading from property graph databases such as Neo4j.
The transformation process shares core principles with relational databases, but adapts to the flexible node-and-edge structure of property graphs.

.. code-block:: python

    from neo4j import GraphDatabase
    from architxt.database.loader.cypher import read_cypher

    with (
        GraphDatabase.driver('bolt://localhost:7687') as driver,
        driver.session() as session,
    ):
        forest = read_cypher(session)

Process Overview
----------------

ArchiTXT transforms a property graph into a hierarchical structure using graph arboricity with a :abbr:`BFS (Breadth-First Search)` strategy.
The process is designed to build compact trees that cover the schema efficiently to ensure tree fit into memory.
Node labels with no incoming edge are selected as roots for the BFS.
These are entry points in the schema graph and cannot be reached from other nodes.

.. important:: Edges are followed as defined in the graph so they should follow the `1-n` cardinality for better distribution.

For each node having a root label, ArchiTXT perform a BFS, recursively traversing edges to build a small, self-contained tree.
Edges are followed only once per BFS to prevents exponential growth of the tree and avoid cycle.
The tree is build during the BFS, where nodes become group tree using the node label as group name and properties become entities.
Edges are represented as relationships between groups.
Attributed edges are promoted as an intermediate node before conversion.
Each group will be assign the node `element_id` as it's :abbr:`OID (Object Identifier)`.

.. important:: If a node has multiple labels, only the first label will be used to define the group name.

Once all node with a root label are exhausted, the process continues for other unvisited nodes following a BFS from previous root.
This ensures that all data is eventually covered without redundancy.

Sampling
^^^^^^^^

It is possible to specify the maximum number of node to read per label to extract a sample of a large database.

.. code-block:: python

    from neo4j import GraphDatabase
    from architxt.database.loader.cypher import read_cypher

    with (
        GraphDatabase.driver('bolt://localhost:7687') as driver,
        driver.session() as session,
    ):
        forest = read_cypher(session, sample=100)  # Only 100 nodes will be extracted for each label

Cyclic Resolution
^^^^^^^^^^^^^^^^^

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
If no BFS root can be determine for that cycle, an arbitrary node label is chosen as the one with the greater cardinality and the maximum number of instance.

.. margin::

    .. mermaid::
        :align: center

        ---
        config:
          theme: neutral
        ---
        graph LR
            Person --> Person

Cycle-edge will yield a relation between the same group.
The nodes can be used as BFS root (ie. it does not count as an input edge).

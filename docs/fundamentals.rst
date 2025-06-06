Fundamentals
============

ArchiTXT represents a database instances as a **hierarchical tree structure** called the meta-model.
This meta-model acts as an intermediary, database-agnostic representation.
It allows flexible transformation from/to various database models such as :

- Relational databases
- Document-oriented databases
- Graph databases (RDF, Property Graph)
- Tabular formats (CSV, TSV, etc.)
- Nested data structure (XML, JSON, YAML, etc.)

Instance Representation
-----------------------

A database instance is a set of :py:class:`~architxt.tree.Tree` organised into a forest.
Each node in these trees represents a fundamental database concept. The key components are:

Entity
    A fundamental data unit representing a name-value couple such as attributes.
    It is represented by `ENT` nodes.

Group
    A collection of **Entities** that share a semantic relationship.
    It is represented by `GROUP` nodes.

Relation
    A connection between two **Groups**.
    It is represented by `REL` nodes.

Collection
    A set of equivalent **Groups** or **Relations**.
    It is represented by `COLL` nodes.

Below is a visual representation of an instance in ArchiTXT:

.. mermaid::
    :alt: Tree representation of the meta model
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD;
        root["ROOT"]
        root --> r1[REL1]
        r1 --> g1[GROUP1]
        r1 --> g2[GROUP2]
        root --> g3[GROUP1]
        g1 --> e1[ENT1]
        g1 --> e2[ENT2]
        g2 --> e3[ENT3]
        g3 --> e4[ENT1]
        g3 --> e5[ENT2]

Schema Definition
-----------------

Given an instance structured as a forest, a :py:class:`~architxt.schema.Schema` is a grammar.
This grammar recognises a subset of trees that compose a valid database instance.

The following, is the minimal schema that recognises the above instance:

.. productionlist::
    ROOT: REL1 | GROUP1 | GROUP2
    REL1: GROUP1 GROUP2
    GROUP1: ENT1 ENT2
    GROUP2: ENT3
    ENT1: <data>
    ENT2: <data>
    ENT3: <data>

Getting Started
===============

.. toctree::
    :hidden:

    getting_started/corpus
    getting_started/cli
    getting_started/ui

ArchiTXT acts as an automated ETL tool that enables the extraction of data from multiple data models and combines them into a unified data repository.
It uses a tree-based structure to represent data in a consistent format (fundamentals/instance-representation), which can then be exported to the output model of your choice; such as a relational database or a property graph.
This allows you to explore multiple databases simultaneously and gain valuable insights from your data.

.. mermaid::
    :alt: ArchiTXT Schema
    :align: center

    ---
    config:
      theme: neutral
    ---
    flowchart TD
        i1@{shape: docs, label: "Textual
            documents"} --> e[Extract];
        i2[(SQL
            Database)] --> e;
        i3[(Graph
            Database)] --> e;
        i4@{shape: docs, label: "Document
            database"} --> e;
        subgraph ArchiTXT
        e --> t[Transform]
        t --> l[Load]
        end
        l --> o[(Unified Data
            Repository)];

        click e "importers.html"
        click t "transformers.html"
        click l "exporters.html"

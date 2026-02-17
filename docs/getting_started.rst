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

**ArchiTXT** is built to work seamlessly with BRAT-annotated corpora that includes pre-labeled named entities.
It can parse the texts using either CoreNLP or SpaCy, depending on your preference and setup.
See the :doc:`importers/text` page for more information.

For CoreNLP, it requires access to a CoreNLP server, which you can set up using the Docker Compose configuration available in the source repository.
To deploy it, you can use the following command:

.. code-block:: bash

    docker compose up -d corenlp

After parsing the annotated texts into **ArchiTXT**'s internal representation, you can infer a database schema and instance based on the annotated entities and generate structured instances accordingly.
See the :doc:`transformers/simplify` page for more information.
After this you can :doc:`exporters` as a relational or property graph database.

**ArchiTXT** is available as a python library but also provides a :doc:`getting_started/cli` for users who prefer working in the terminal.
You can run the CLI using:

.. code-block:: bash

    architxt --help

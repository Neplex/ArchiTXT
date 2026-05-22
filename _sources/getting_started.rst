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

Text to Database Pipeline
-------------------------

**ArchiTXT** is built to work seamlessly with BRAT-annotated corpora that includes pre-labeled named entities.
It can parse the texts using either CoreNLP or SpaCy, depending on your preference and setup.
See the :doc:`importers/text` page for more information.

This guide demonstrates the complete workflow from raw text to a structured database export.

Setup a Parser
""""""""""""""

ArchiTXT needs a constituency parser to process text. You have two main options:

* **CoreNLP**: A robust Java-based parser. Recommended for the CLI.
* **Benepar**: A Python-based parser using SpaCy. Recommended for Python scripts.

See the :doc:`importers/text` page for more information.

For CoreNLP, it requires access to a CoreNLP server, which you can set up using the Docker Compose configuration available in the source repository.
The easiest way to run CoreNLP is via Docker:

.. code-block:: bash

    docker compose up -d corenlp

Load and Parse the Corpus
"""""""""""""""""""""""""

Use the `load corpus` command to parse your text files. This command reads the text (BRAT annotated or raw), performs NLP processing, and stores the result in ArchiTXT's internal format.

.. code-block:: bash

    architxt load corpus data/my_corpus/ --output my_database.fs

.. note::
  The CLI currently defaults to CoreNLP. To use Benepar, refer to the Python API section below.

Simplify the Structure
""""""""""""""""""""""

After parsing the annotated texts into **ArchiTXT**'s internal representation, you can infer a database schema and instance based on the annotated entities and generate structured instances accordingly.

.. code-block:: sh

    $ architxt simplify my_database.fs --tau 0.7 --epoch 100

See :doc:`transformers/simplification` for more details on simplification strategies.

Export to Database
""""""""""""""""""

Finally, export your data to your desired format (see :doc:`exporters` for more details).

.. tab-set::

    .. tab-item:: SQL

        .. code-block:: sh

            $ architxt export sql my_database.fs --uri sqlite:///output.db

    .. tab-item:: Neo4j (Graph)

        .. code-block:: sh

            $ architxt export graph my_database.fs --uri bolt://localhost:7687 --password mypassword


SQL to Graph Migration
----------------------

ArchiTXT can also migrate data between database paradigms.

Import from SQL
"""""""""""""""

.. code-block:: sh

    $ architxt load sql postgresql://user:pass@localhost/mydb --output mydb.fs

Export to Graph
"""""""""""""""

.. code-block:: sh

    $ architxt export graph mydb.fs --uri bolt://localhost:7687 --password mypassword

Python API & Alternative Parsers
--------------------------------

You can use ArchiTXT within your Python scripts, which gives you flexibility to use different parsers like Benepar.

.. code-block:: python

    from architxt.nlp.parser.benepar import BeneparParser
    from architxt.nlp import raw_load_corpus
    from architxt.bucket.zodb import ZODBTreeBucket
    from architxt.transformers.simplification import rewrite
    from architxt.schema import Schema
    import anyio

    async def main():
        # Initialize Benepar Parser
        parser = BeneparParser(spacy_models={'English': 'en_core_web_sm'})

        # Load corpus
        trees = await raw_load_corpus(
            ['data/corpus.txt'],
            ['English'],
            parser=parser
        )

        with ZODBTreeBucket(storage_path="my_database.fs") as forest:
            forest.async_update(trees, commit=True)
            print(f"Loaded {len(forest)} trees.")

            # Structure the data
            rewrite(forest, tau=.7, epoch=20)

            # Display the schema
            schema = Schema.from_forest(forest)
            print(schema)

    if __name__ == "__main__":
        anyio.run(main)

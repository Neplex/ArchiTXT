Getting Started with ArchiTXT UI
================================

ArchiTXT provides a web-based user interface built with Streamlit, allowing for interactive exploration of your corpora, schema inference, and data simplification.
This UI is particularly useful for visualizing the transformation process and fine-tuning parameters without writing code.

Launching the Application
-------------------------

The ArchiTXT UI is accessible via a dedicated Command Line Interface (CLI). To start the server, run:

.. code-block:: bash

    $ architxt ui

By default, the application will be available at ``http://localhost:8501``.

Core Interface Overview
-----------------------

The interface is divided into several functional areas designed to guide you through the data lifecycle:

1. **Global Metrics**: Located at the top of the page, this section provides real-time counts of your current Forest (Total Trees, Entities, Groups, and Relations).
2. **Navigation**: Use the top navigation bar to switch between **File** operations (Import/Export) and **Tools** (Visualize/Transform).

Data Lifecycle Workflows
------------------------

The UI works on one forest at a time, allowing you to iteratively refine your data.
The metrics at the top will update dynamically as you import new data and apply transformations.
The **clear data** button allows you to reset the current session and start fresh with a new dataset.
The typical workflow involves:

Importing Data
^^^^^^^^^^^^^^

Navigate to **File > Import** to load data into the ArchiTXT environment. The UI supports five primary sources:

* **Text Corpus**: Process raw text files using NLP parsers (CoreNLP or Benepar). You can configure entity resolution (UMLS, MeSH, etc.) and filter specific relations.
* **SQL/Graph Databases**: Connect to relational databases (via SQLAlchemy URI) or Neo4j graph instances.
* **Document Files**: Support for structured formats including ``.json``, ``.toml``, ``.csv``, and ``.xlsx``.
* **JSONL File**: Direct import of ArchiTXT native JSON Lines format.

Visualization
^^^^^^^^^^^^^

Navigate to **Tools > Visualize** to inspect your data:

* **Schema View**: A dynamic graph visualization showing the relationships between entities and groups.
* **Instance Explorer**: Select specific groups to view their underlying data in a searchable tabular format.

Transformation & Simplification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to **Tools > Transform** to refine your data structures. Two methods are available:

* **Rule-Based**: Uses similarity (Tau) and support thresholds to merge or rewrite tree structures.
* **LLM-Based**: Leverages Large Language Models (via local HuggingFace models or APIs like OpenAI) to perform intelligent rewriting and refinement.

Exporting Results
^^^^^^^^^^^^^^^^^

Once processed, navigate to **File > Export** to save your session. You can export back to:

* SQL Databases
* Graph Databases (Neo4j)
* JSONL files (available for direct browser download)

Loading textual datas
=====================

ArchiTXT enables seamless integration of textual data by representing it as a hierarchical tree structure, derived from the syntax tree.
Through its built-in simplification algorithm, ArchiTXT organizes this data into structured instances, which can be easily combined with existing database instances.
All tools for manipulating textual data are available in the :py:mod:`architxt.nlp` module.

.. seealso::

    :doc:`../getting_started/corpus`
        More information on BRAT format support.

    :doc:`../examples/corpus_exploration`
        A real-case usage examples.


Annotations
-----------

While ArchiTXT does not include built-in data extraction tools, it operates on text that is accompanied by annotations.
These annotations can be directly provided to ArchiTXT or loaded from corpora formatted in the BRAT format, packaged as a zip or tar-archived folder.

Supported annotation types include:

Named Entities
    Specific segments of text representing objects or concepts with distinct names or types. For example, the sentence "Alice likes apples", "Alice" can be annotated as a Person entity, and "apples" as a Fruit entity.

Relations
    Binary connections between two entities that express semantic relationships. In "Alice likes apples", the word "likes" indicates a possible relation between the Person entity "Alice" and the Fruit entity "apples".

Named entities are interpreted as entities within the ArchiTXT meta-model.

Text to trees
-------------

Consider the example sentence:

    The heart rate was 100 bpm

With the following named entities:

- *"heart rate"* annotated as a **Sign or Symptom (SOSY)**
- *"100"* annotated as a **Value**
- *"bpm"* annotated as a **Unit**

ArchiTXT start by extracting the syntax tree of each text.
The raw syntax tree for the example sentence is as follows:

.. mermaid::
    :alt: Syntax tree
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> NP1["NP"]
        S --> VP
        NP1 --> DT
        NP1 --> NN1["NN"]
        NP1 --> NN2["NN"]
        DT --> The
        NN1 --> heart
        NN2 --> rate
        VP --> VBD
        VBD --> was
        VP --> NP2["NP"]
        NP2 --> CD
        CD --> 100
        NP2 --> NN3["NN"]
        NN3 --> bpm

Entities are then embedded into the tree:

.. mermaid::
    :alt: Syntax tree with entities
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> NP1["NP"]
        S --> VP
        NP1 --> DT
        DT --> The
        NP1 --> sosy["ENT SOSY"]
        sosy --> NN1["NN"]
        sosy --> NN2["NN"]
        NN1 --> heart
        NN2 --> rate
        VP --> VBD
        VBD --> was
        VP --> NP2["NP"]
        NP2 --> CD
        CD --> value["ENT VALUE"]
        value --> 100
        NP2 --> NN3["NN"]
        NN3 --> unit["ENT UNIT"]
        unit --> bpm

The tree is then simplified by removing unnecessary branches, focusing only on meaningful entities:

.. mermaid::
    :alt: Simplified syntax tree
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> NP1["NP"]
        S --> VP
        NP1 --> sosy["ENT SOSY"]
        sosy --> NN1["NN"]
        sosy --> NN2["NN"]
        NN1 --> heart
        NN2 --> rate
        VP --> NP2["NP"]
        NP2 --> CD
        CD --> value["ENT VALUE"]
        value --> 100
        NP2 --> NN3["NN"]
        NN3 --> unit["ENT UNIT"]
        unit --> bpm

Finally, the tree is reduced by eliminating unnecessary nodes:

.. mermaid::
    :alt: Final syntax tree
    :align: center

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> sosy["ENT SOSY"]
        sosy --> heart
        sosy --> rate
        S --> VP
        VP --> value["ENT VALUE"]
        value --> 100
        VP --> unit["ENT UNIT"]
        unit --> bpm


Parse your documents
--------------------

ArchiTXT provides multiple parsing backends to process documents and extract structured representations.

CoreNLP
+++++++

ArchiTXT can use `CoreNLP <https://stanfordnlp.github.io/CoreNLP/>`_ to process the documents.
To use this, you need to have a CoreNLP server running with the appropriate language model installed.

.. note::

    A pre-configured Docker setup for a CoreNLP server (supporting both English and French) is available in the GitHub repository.
    This setup is intended for development use only.
    For production deployment, consult the `official CoreNLP documentation <https://stanfordnlp.github.io/CoreNLP/corenlp-server.html>`_.

To initialize a :py:class:`architxt.nlp.parser.corenlp.CoreNLPParser` in ArchiTXT, use:

.. code-block:: python

    from architxt.nlp.parser.corenlp import CoreNLPParser

    parser = CoreNLPParser(corenlp_url='http://localhost:9000')

Benepar
+++++++

ArchiTXT also supports the `Benepar parser <https://github.com/nikitakit/self-attentive-parser>`_, which integrates with `SpaCy <https://spacy.io>`_ for syntactic parsing.
To initialize a :py:class:`architxt.nlp.parser.benepar.BeneparParser`, use:

.. code-block:: python

    from architxt.nlp.parser.benepar import BeneparParser

    parser = BeneparParser(spacy_models={
        'English': 'en_core_web_md',
        'French': 'fr_core_news_md',
    })

You need to specify the SpaCy models to use for each language, and they must be installed beforehand.
SpaCy provides various models (`sm`, `md`, `lg`) with different sizes and capabilities.
You can install them using:

.. code-block:: bash

    python -m spacy download en_core_web_md
    python -m spacy download fr_core_news_md

For a full list of available models, visit the `SpaCy model directory <https://spacy.io/models>`_.

Loading textual datas
=====================

ArchiTXT enables the seamless integration of textual data by representing it as a hierarchical tree structure, based on the syntax tree.
Through its simplification algorithm, ArchiTXT organizes this data into structured instances.
They can then be effortlessly combined with existing database instances.

Text transformation
-------------------

While ArchiTXT does not include built-in data extraction tools, it operates on text accompanied by annotations.
Annotations can be directly provided to ArchiTXT, and it also supports loading corpora formatted in the BRAT format, packaged as a tar-archived folder.
It supports:

- **Named Entities**: Specific segments of text representing objects or concepts with distinct names or types. For example, in the sentence "Alice likes apples," "Alice" can be annotated as a Person entity, and "apples" as a Fruit.

- **Relations**: Binary connections between two entities. In the previous example, "likes" indicates a possible relation between "Alice" and "apples".

Named entities are interpreted as entities in the meta-model.



Text enrichment
---------------

Consider the sentence "The heart rate was 100 bpm" with the following named entities:

- "heart rate" as sign or symptom (SOSY)
- "100" as value
- "bpm" as unit

ArchiTXT utilizes a CoreNLP server to obtain the syntax tree of each text before enriching them with entities and relations.
From the example above we obtain the following syntax tree :

.. mermaid::
    :alt: Syntax tree

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

Entities are then incorporated into the tree:

.. mermaid::
    :alt: Syntax tree

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> NP1["NP"]
        S --> VP
        NP1 --> DT
        DT --> The
        NP1 --> sosy["ENTITY SOSY"]
        sosy --> NN1["NN"]
        sosy --> NN2["NN"]
        NN1 --> heart
        NN2 --> rate
        VP --> VBD
        VBD --> was
        VP --> NP2["NP"]
        NP2 --> CD
        CD --> value["ENTITY VALUE"]
        value --> 100
        NP2 --> NN3["NN"]
        NN3 --> unit["ENTITY UNIT"]
        unit --> bpm

The tree is then simplified by removing branches that do not contain entities:

.. mermaid::
    :alt: Syntax tree

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> NP1["NP"]
        S --> VP
        NP1 --> sosy["ENTITY SOSY"]
        sosy --> NN1["NN"]
        sosy --> NN2["NN"]
        NN1 --> heart
        NN2 --> rate
        VP --> NP2["NP"]
        NP2 --> CD
        CD --> value["ENTITY VALUE"]
        value --> 100
        NP2 --> NN3["NN"]
        NN3 --> unit["ENTITY UNIT"]
        unit --> bpm

Finally, the tree is reduced by eliminating unnecessary nodes:

.. mermaid::
    :alt: Syntax tree

    ---
    config:
      theme: neutral
    ---
    graph TD;
        S --> sosy["ENTITY SOSY"]
        sosy --> heart
        sosy --> rate
        S --> VP
        VP --> value["ENTITY VALUE"]
        value --> 100
        VP --> unit["ENTITY UNIT"]
        unit --> bpm

.. seealso::

    - :doc:`../getting_started/corpus` for more information on BRAT format support.
    - :doc:`../examples/corpus_exploration` for real-case usage examples.

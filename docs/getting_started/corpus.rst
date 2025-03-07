Preparing a Text Corpora
========================

ArchiTXT requires your text corpora to be in a Gzipped Tar archive using the BRAT annotation format.
In this format, the text and annotations are stored in two files with the same name but different extensions (.txt for text and .ann for annotations).
For example, `doc-123.txt` contains the text, while `doc-123.ann` holds the annotations (named entities, relations, events, etc.).
Your archive should look similar to this :

.. code-block:: text

    corpus.tar.gz
    ├── doc-123.txt
    ├── doc-123.ann
    ├── doc-124.txt
    ├── doc-124.ann
    ├── doc-125.txt
    ├── doc-125.ann
    └── ...

.. important::

    1. **One Sentence per Line:** You should have one sentence per line in your text file.
    2. **Non-Overlapping Annotations:** The start and end offsets for each annotated entity should not overlap with any other entity in the same sentence. Overlapping spans are not yet handle and will be ignored by ArchiTXT.


BRAT Annotation Format
----------------------

Annotations in BRAT are defined in a simple, tab-delimited format.
Each annotation line follows this structure:

.. code-block::

    T<ID> <Entity_Type> <Start_Offset> <End_Offset> <Entity_Text>

- **T<ID>:** A unique identifier for the annotation (e.g., T1, T2, etc.).
- **Entity_Type:** The category or type of the entity (e.g., `Person`, `Location`, `Animal`).
- **Start_Offset and End_Offset:** The character positions in the sentence where the entity begins and ends.
- **Entity_Text:** The exact text span that has been annotated.

Consider the following sentence in your text file (`document.txt`):

.. code-block::

    The quick brown fox jumps over the lazy dog.

An accompanying annotation file (`document.ann`) might look like:

.. code-block::

    T1 Animal 16 19 fox
    T2 Animal 35 39 dog

In this example:

- **T1** annotates the entity "fox" starting at character position 16 and ending at 19.
- **T2** annotates the entity "dog" starting at position 35 and ending at 39.
- Both annotations are non-overlapping and correspond to entities in the sentence.

.. seealso::

    `BRAT Rapid Annotation Tool website <https://brat.nlplab.org/standoff.html>`_ for more information about the standoff format

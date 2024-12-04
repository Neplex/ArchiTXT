Preparing a Textual Corpora for ArchiTXT
=========================================

ArchiTXT requires your textual corpora to be prepared using a format based on the BRAT Rapid Annotation Tool.
In particular, each sentence should be placed on a separate line and entity annotations must not overlap.

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

Example
-------

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

Guidelines for Annotation
-------------------------

1. **One Sentence per Line:**
   Ensure that every sentence is isolated on its own line in your text file. This avoids multi-line sentence complications and makes processing straightforward.

2. **Distinct, Non-Overlapping Annotations:**
   Make sure that the start and end offsets for each annotated entity do not overlap with any other entity in the same sentence. Overlapping spans are not yet handle and will be ignored by ArchiTXT.

Additional Resources
--------------------

For more information on the BRAT Rapid Annotation Tool and its annotation format, please visit the `BRAT Rapid Annotation Tool website <http://brat.nlplab.org/>`_.

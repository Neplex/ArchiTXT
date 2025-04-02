Getting Started
===============

.. toctree::
    :hidden:

    getting_started/corpus
    getting_started/cli
    getting_started/ui

.. mermaid::
    :alt: ArchiTXT Schema

    ---
    config:
      theme: neutral
    ---
    flowchart LR
        i1@{shape: docs, label: "Documents"} --> e[Extract];
        i2[(SQL
            Database)] --> e;
        i3[(Graph
            Database)] --> e;
        subgraph ArchiTXT
        e --> t[Transform]
        t --> l[Load]
        end
        l --> o[(Unified Data
            Repository)];

        click e "load_data.html"
        click t "transform_data.html"
        click l "export_data.html"

Prepare Your Text Corpus
------------------------

Before you start, your corpora must be in the BRAT annotation format and archived as a Gzipped Tar file.

.. note::

    For full instructions on preparing your corpus, please refer to :doc:`getting_started/corpus` page.

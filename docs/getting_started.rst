Getting Started
===============

.. toctree::
    :hidden:

    getting_started/corpus
    getting_started/cli
    getting_started/ui

Prepare Your Text Corpus
------------------------

Before you start, your corpora must be in the BRAT annotation format and archived as a Gzipped Tar file.

.. note::

    For full instructions on preparing your corpus, please refer to :doc:`getting_started/corpus` page.


CoreNLP sever
-------------

ArchiTXT rely on `CoreNLP <https://stanfordnlp.github.io/CoreNLP/>`_ to process the documents.
You need to have a CoreNLP server up and running with the needed language model installed.

.. note::

    A Docker configuration for a CoreNLP server, which includes both English and French models, is available in the GitHub repository.
    However, this setup is intended for development purposes only and is not recommended for production use.
    For proper deployment, please refer to the `official documentation <https://stanfordnlp.github.io/CoreNLP/corenlp-server.html>`_ for guidance on setting it up on your system.

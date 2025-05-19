Loading document databases
==========================

.. margin::

    .. mermaid::
        :caption: Document conversion process
        :align: center

        ---
        config:
          theme: neutral
        ---
        flowchart TD
            A@{shape: docs, label: "Document Files"} -->|read| B[Raw Data]
            B -->|parse| C[Raw Tree]
            C -->|transform| D[ArchiTXT Tree]

ArchiTXT supports loading document databases (such as JSON, XML, YAML, TOML, CSV, and Excel) through the :py:mod:`architxt.database.loader.documents` module.
These documents are converted into data :py:class:`~architxt.tree.Tree` corresponding the metamodel.

The document-to-tree conversion process involves three steps:

#. **Read**: Detect and parse the input file into a native Python nested structure compose of either :py:obj:`dict` or :py:obj:`list`.
#. **Parse**: Convert the Python structure into a :py:class:`~architxt.tree.Tree` composed of `COLL`, `GROUP`, and `ENT` nodes.
#. **Transform**: Optionally extract relationships implied by nested groups, transforming the tree to align it with the target metamodel.

The decomposition in three steps enables parsing not only supported data formats but also arbitrary Python data structures.
The resulting raw trees are not considered valid on their own but can be combined with syntax trees before applying more advanced structuring algorithms.

.. warning::

    The transformation described here is specifically designed for tree-like data.
    Applying it to arbitrary or improperly structured trees may result in invalid or incoherent outputs.

Parsing nested data structures
------------------------------

The parsing process is performed via :py:func:`~architxt.database.loader.documents.read_tree`.
This function traverses nested Python structures and constructs a corresponding :py:class:`~architxt.tree.Tree` based on the following rules:

- A :py:obj:`dict` becomes a `GROUP` node, where each key/value pair is parsed into a subtree.
- A :py:obj:`list` becomes a `COLL` node, where each element is parsed into a subtree.
- A scalar value (e.g., :py:obj:`str`, :py:obj:`int`, :py:obj:`float`, :py:obj:`bool`) becomes an `ENT` node wrapping the value.

.. dropdown:: Example

    Consider the following JSON document:

    .. code-block:: json
        :caption: An example of a JSON document

        [
            {
                "userId": 1,
                "username": "johndoe",
                "profile": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "birthDate": "1990-01-01"
                }
            }
        ]

    This input is converted into the following tree structure:

    .. mermaid::
        :alt: JSON example as a raw tee
        :align: center

        ---
        config:
          theme: neutral
        ---
        graph TD
            users["COLL users"]

            users --> user["GROUP user"]
            user --> userId["ENT userId"] --> userIdVal["1"]
            user --> username["ENT username"] --> usernameVal["johndoe"]

            user --> profile["GROUP profile"]
            profile --> firstName["ENT firstName"] --> firstNameVal["John"]
            profile --> lastName["ENT lastName"] --> lastNameVal["Doe"]
            profile --> birthDate["ENT birthDate"] --> birthDateVal["1990-01-01"]

Transforming Raw Trees
----------------------

Once a raw tree is constructed, it can be transformed into a flattened structure aligned with the metamodel using :py:func:`~architxt.database.loader.documents.parse_document_tree`.

This transformation:

- Converts nested `GROUP` nodes into `REL` nodes, establishing explicit relationships between parent and child subtrees.
- Duplicates the parent node for each nested group while retaining only its direct `ENT` children as part of the `GROUP`.
- If the root of the raw tree is a `COLL`, the transformation produces a forest; constructing one tree per collection element.

.. dropdown:: Example

    Given the raw tree from the previous example, the transformation produces the following structure that conforms to the ArchiTXT metamodel:

    .. mermaid::
        :alt: JSON example converted to ArchiTXT meta-model
        :align: center

        ---
        config:
          theme: neutral
        ---
        graph TD
            root["ROOT"]

            root --> coll["COLL user<->profile"]
            coll --> rel["REL user<->profile"]

            rel --> user["GROUP user"]
            user --> userId["ENT userId"] --> userIdVal["1"]
            user --> username["ENT username"] --> usernameVal["johndoe"]

            rel --> profile["GROUP profile"]
            profile --> firstName["ENT firstName"] --> firstNameVal["John"]
            profile --> lastName["ENT lastName"] --> lastNameVal["Doe"]
            profile --> birthDate["ENT birthDate"] --> birthDateVal["1990-01-01"]

Supported File Formats
----------------------

ArchiTXT supports a wide range of document formats through pluggable parsers.
Each format is handled by a specific backend parser:

.. hlist::
    :columns: 2

    - **JSON**: :py:func:`json.load`
    - **TOML**: :py:func:`toml.loads`
    - **YAML**: :py:meth:`ruamel.yaml.YAML.load_all`
    - **XML**: :py:func:`xmltodict.parse`
    - **CSV**: :py:func:`pandas.read_csv`
    - **Excel**: :py:func:`pandas.read_excel`

.. important::

    Parsers are applied in order; if none succeed, a :py:exc:`ValueError` is raised.

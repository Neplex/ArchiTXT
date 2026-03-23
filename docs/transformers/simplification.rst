Simplification
==============

ArchiTXT provides powerful tools to simplify the structure of your database.
Simplification iteratively rewrites the instance to discover and uniformize latent structures (Groups, Relations).
It can be used to discover a schema from text or automatically merge multiple datasource together (see :doc:`../examples` for more examples).

There are two main approaches to simplification in ArchiTXT:

1.  **Rule-Based Simplification**: Uses statistical analysis and predefined tree rewriting operations.
2.  **LLM-Based Simplification**: Leverages Large Language Models to semantically simplify the schema.

Rule-Based Simplification
-------------------------

The simplification algorithm operates iteratively (defined by ``epoch``). In each iteration, it performs the following steps:

1.  **Clustering**: Subtrees within the forest are clustered based on their structural and semantic similarity. The similarity is controlled by the ``tau`` (threshold) and ``decay`` parameters.
2.  **Pattern Discovery**: Frequent patterns in these clusters (occurring more than ``min_support`` times) are identified as potential candidates for structural abstraction (e.g., forming a Group).
3.  **Rewriting**: A sequence of **Edit Operations** is applied to the trees. These operations transform the tree structure based on the discovered patterns.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: sh

            $ architxt simplify my_database.fs --tau=0.7 --decay=3 --epoch=100

    .. tab-item:: Python

        .. code-block:: python

            from architxt.bucket.zodb import ZODBTreeBucket
            from architxt.simplification.tree_rewriting import rewrite
            import anyio

            async def main():
                with ZODBTreeBucket(storage_path="my_database.fs") as forest:
                    rewrite(forest, tau=.7, decay=3, epoch=100)

            if __name__ == "__main__":
                anyio.run(main)

Key Parameters
^^^^^^^^^^^^^^

Tau (τ)
    The similarity threshold for clustering. A value between 0 and 1. Higher values mean subtrees must be more similar to be grouped together.
    * High τ (e.g., 0.9): Conservative clustering, leads to more specific groups.
    * Low τ (e.g., 0.4): Aggressive clustering, leads to more generic groups.

Decay (δ)
    Controls how much the context (surrounding nodes) influences the similarity calculation.
    * Higher decay reduces the influence of distant nodes.

Min Support
    The minimum number of occurrences required for a pattern to be considered significant. Patterns appearing less frequently are ignored.

Edit Operations
^^^^^^^^^^^^^^^

The simplification process applies a sequence of operations. The standard operations include:

*   **FindSubGroups**: Identifies sets of entities that frequently appear together and groups them.
*   **MergeGroups**: Merges similar groups that might have been identified separately.
*   **FindRelations**: Identifies relationships between groups.
*   **FindCollections**: Identifies lists or sets of similar items (Groups or Relations).
*   **ReduceTop/ReduceBottom**: Prunes unnecessary levels of the tree hierarchy.

LLM-Based Simplification
------------------------

The `simplify-llm` command uses an LLM to identify and merge semantically equivalent structures that might be structurally different. This is computationally more intensive but can yield better results for complex schemas.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: sh

            $ architxt simplify-llm my_database.fs --model HuggingFaceTB/SmolLM2-135M-Instruct --max-tokens=8000

    .. tab-item:: Python

        .. code-block:: python

            from architxt.bucket.zodb import ZODBTreeBucket
            from architxt.simplification.llm import estimate_tokens, llm_rewrite
            from langchain.chat_models import init_chat_model
            import anyio

            async def main():
                llm = init_chat_model("HuggingFaceTB/SmolLM2-135M-Instruct", max_tokens=8_000)

                with ZODBTreeBucket(storage_path="my_database.fs") as forest:
                    await llm_rewrite(forest, llm, max_tokens=8_000)

            if __name__ == "__main__":
                anyio.run(main)

Key Parameters
^^^^^^^^^^^^^^

Model
    The identifier of the LLM model to use (e.g., from HuggingFace).

Max tokens
    Maximum number of token to use per prompt.

Refining steps
    Number of refinement steps to perform.

.. note::
    LLM simplification requires additional dependencies. Install them with `pip install architxt[llm]`.

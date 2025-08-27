import itertools
import json
import warnings
from collections import Counter
from collections.abc import Collection, Generator, Iterable

import mlflow
import more_itertools
import torch
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm.auto import tqdm, trange

from architxt.bucket import TreeBucket
from architxt.metrics import Metrics
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC
from architxt.tree import Forest, NodeType, Tree, has_type
from architxt.utils import windowed_shuffle

__all__ = ['llm_rewrite']

DEFAULT_PROMPT = PromptTemplate.from_template("""
You are tasked with standardizing a list of tree by identifying semantic nodes, naming them, and removing non-semantic ones.

Hard requirements (do not omit):
- Each tree start by a single top-level node:
   {{
     "oid": "...",
     "name": "ROOT",
     "type": null,
     "metadata": {{}},
     "children": [...]
   }}
- Allowed node types: GROUP, REL, ENT and every ENT should be in a GROUP.
- Untyped nodes should be identified as GROUP/REL candidates or removed from the structure moving their children to it's parent.
- You can introduce new node or duplicates subtrees if needed to express the semantics.
- Preserve ENT nodes and their leaves exactly; do not alter names or leaf values.
- Preserve the semantic of the tree structure, do not group ENT that are far apart for example.
- GROUP creation:
   * Parent with more than two distinct ENT children should be considered as a GROUP.
   * Must have at least 2 ENT children; no duplicate ENT names.
   * Name GROUPs based on the semantic meaning of the group (Employee, Treatment, etc).
- REL creation:
   * Parent connecting exactly 2 distinct GROUPs becomes a REL.
   * Name RELs based on the semantic meaning of the relation (WorksAt, Treat, etc).
- Attempt to minimize the number of distinct GROUP and REL name types across all trees by reusing names. {vocab}
- Remove untyped/non-semantic nodes after forming GROUPs and RELs.
- Every node must have the shape: {{"oid": <string|null>, "name": <string>, "type": <"GROUP"|"REL"|"ENT"|null>, "metadata": <object|null>, "children": <array of nodes|string>}}
   * If you create a new node, set "oid": null.
   * Leaf values for ENT nodes are plain strings and must be preserved exactly.
   * Preserve every existing oid. Only new nodes get oid: null. Do not invent OIDs.
- Output only the final JSON structure as a numbered list, with one element per line in the same order as the input, eg:
    1. {{"oid": "...", "name": "ROOT", "type": "GROUP", "metadata": {{}}, "children": [...]}}
    2. {{"oid": "...", "name": "ROOT", "type": "GROUP", "metadata": {{}}, "children": [...]}}

Process:
1) Parse input trees.
2) Bottom-up pass: detect GROUP candidates
3) Second pass: detect RELs
4) Flatten/remove untyped nodes that remain non-semantic, moving children up

Example:
Input:
1. {{"oid":"1","name":"UNDEF","type":null,"children":[{{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]}},{{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}}]}}
Output (numbered-line):
1. {{"oid":"1","name":"Fruit","type":"GROUP","children":[{{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]}},{{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}}]}}

Input trees:
{trees}
""")


def _tree_to_list(trees: Iterable[Tree]) -> str:
    return '\n'.join(f'{i}. {tree.to_json()}' for i, tree in enumerate(trees, start=1) if isinstance(tree, Tree))


def _parse_tree_output(raw_output: str | None, *, fallback: Tree, debug: bool = False) -> Tree:
    """Try parsing raw LLM output into a Tree, fallback if parsing fails."""
    if not raw_output:
        return fallback

    try:
        json_data = json.loads(raw_output)

        if isinstance(json_data, dict):
            return Tree.from_json(json_data)

        if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
            return Tree.from_json(json_data[0])

    except ValueError:
        if debug:
            warnings.warn(f'Failed to parse tree "{raw_output}"', RuntimeWarning)

    return fallback


@torch.inference_mode()
def llm_simplify(
    llm: BaseLLM,
    max_token: int,
    prompt: PromptTemplate,
    trees: Iterable[Tree],
    debug: bool,
    vocab: Collection[str] | None = None,
) -> Generator[Tree, None, None]:
    """
    Simplify parse trees using an LLM.

    :param llm: LLM model to use.
    :param max_token: Maximum number of tokens to allow per prompt.
    :param prompt: Prompt template to use.
    :param trees: Sequence of trees to simplify.
    :param debug: Whether to enable debug logging.
    :param vocab: Optional list of vocabulary words to use in the prompt.

    :yield: Simplified trees objects with the same oid as input.
    """
    vocab_str = f"Prefer using names from this vocabulary: {', '.join(vocab)}." if vocab else ""
    prompt = prompt.partial(vocab=vocab_str)
    chain = prompt | llm | NumberedListOutputParser()

    def count_tokens(documents: Iterable[Tree]) -> int:
        full_prompt = prompt.format(trees=_tree_to_list(documents))
        return llm.get_num_tokens(full_prompt)

    for batch in more_itertools.constrained_batches(
        trees,
        max_size=max_token,
        get_len=count_tokens,
        strict=False,
    ):
        llm_results = chain.invoke({'trees': _tree_to_list(batch)})

        for tree, simplified in itertools.zip_longest(batch, llm_results):
            tree = _parse_tree_output(simplified, fallback=tree, debug=debug)
            if tree:
                yield tree


def get_vocab(forest: Iterable[Tree], min_support: int) -> tuple[str, ...]:
    vocab_counter = Counter(
        subtree.label
        for tree in forest
        for subtree in tree.subtrees(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL}))
    )
    return tuple(label for label, _ in vocab_counter.most_common(min_support))


def llm_rewrite(
    forest: Forest,
    *,
    llm: BaseLLM,
    max_token: int,
    tau: float = 0.7,
    min_support: int | None = None,
    refining_steps: int = 0,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    prompt: PromptTemplate = DEFAULT_PROMPT,
    debug: bool = False,
) -> Metrics:
    """
    Rewrite a forest into a valid schema using a LLM agent.

    :param forest: A forest to be rewritten in place.
    :param llm: The LLM model to interact with for rewriting and simplification tasks.
    :param max_token: The token limit of the prompt.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support for vocab.
    :param refining_steps: Number of refining steps to perform after the initial rewrite.
    :param metric: The metric function used to compute similarity between subtrees.
    :param prompt: The prompt template to use for the LLM during the simplification.
    :param debug: Whether to enable debug logging.

    :return: A `Metrics` object encapsulating the results and metrics calculated for the LLM rewrite process.
    """
    metrics = Metrics(forest, tau=tau, metric=metric)
    min_support = min_support or max((len(forest) // 20), 2)

    if mlflow.active_run():
        mlflow.log_params(
            {
                'nb_sentences': len(forest),
                'tau': tau,
                'min_support': min_support,
                'metric': metric.__name__,
                'refining_steps': refining_steps,
            }
        )
        metrics.log_to_mlflow(0, debug=debug)

    for _ in trange(refining_steps + 1, leave=False):
        vocab = get_vocab(forest, min_support)

        simplification = tqdm(windowed_shuffle(forest), leave=False, total=len(forest))
        simplification = llm_simplify(llm, max_token, prompt, simplification, debug, vocab)

        if isinstance(forest, TreeBucket):
            forest.update(simplification)

        else:
            forest[:] = list(simplification)

    metrics.update()

    if mlflow.active_run():
        metrics.log_to_mlflow(1, debug=debug)

    return metrics

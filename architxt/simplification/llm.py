import itertools
import json
import warnings
from collections import Counter
from collections.abc import Collection, Generator, Iterable, Sequence
from contextlib import nullcontext

import json_repair
import mlflow
import more_itertools
import torch
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from tqdm.auto import tqdm, trange

from architxt.bucket import TreeBucket
from architxt.metrics import Metrics
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC
from architxt.tree import Forest, NodeType, Tree, has_type
from architxt.utils import windowed_shuffle

__all__ = ['llm_rewrite']

DEFAULT_PROMPT = PromptTemplate.from_template("""
Restructure JSON trees into a single uniform schema. Edit nodes by Add/Remove/Move/Rename.
ENT = property, GROUP = table, REL = relation.

Node format:
{{"oid":<str|null>,"name":<str>,"type":"GROUP"|"REL"|"ENT"|null,"metadata":<obj|null>,"children":[...]}}

Rules:
- Do NOT modify, rename or add new ENT nodes; keep them unchanged!
- You can duplicates ENT nodes if needed.
- Return one simplified tree per input. No notes or explanations.
- Each output tree must start with root:
  {{"oid":null,"name":"ROOT","type":null,"metadata":{{}},"children":[...]}}
- Create meaningful GROUP nodes to collect related ENT nodes.
- Link GROUPs with REL nodes where appropriate.
- Preserve original oids; any new node gets "oid":null.
{vocab}

Example input:
1. {{"oid":"1","name":"UNDEF","type":null,"children":[{{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]}},{{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}}]}}
2. {{"oid":"4","name":"UNDEF","type":null,"children":[{{"oid":"5","name":"FruitName","type":"ENT","children":["orange"]}},{{"oid":"6","name":"PersonName","type":"ENT","children":["Alice"]}},{{"oid":"7","name":"Age","type":"ENT","children":["30"]}}]}}

Example output:
1. {{"oid":null,"name":"ROOT","type":null,"children":[{{"oid":"1","name":"Fruit","type":"GROUP","children":[{{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]}},{{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}}]}}]}}
2. {{"oid":null,"name":"ROOT","type":null,"children":[{{"oid":null,"name":"Eat","type":"REL","children":[{{"oid":null,"name":"Fruit","type":"GROUP","children":[{{"oid":"5","name":"FruitName","type":"ENT","children":["orange"]}}]}},{{"oid":null,"name":"Person","type":"GROUP","children":[{{"oid":"6","name":"PersonName","type":"ENT","children":["Alice"]}},{{"oid":"7","name":"Age","type":"ENT","children":["30"]}}]}}]}}]}}

Now normalize these trees:
{trees}
""")


def _tree_to_list(trees: Iterable[Tree]) -> str:
    return '\n'.join(
        f'{i}. {json.dumps(tree.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)}'
        for i, tree in enumerate(trees, start=1)
        if isinstance(tree, Tree)
    )


def _validate(tree: Tree) -> Tree:
    """Validate tree structure, removing invalid groups and relations."""
    for rel in tree.subtrees(lambda x: has_type(x, NodeType.REL)):
        if len(rel) != 2 or not all(has_type(child, NodeType.GROUP) for child in rel):
            rel.label = f'UNDEF_{rel.oid.hex}'

    for group in tree.subtrees(lambda x: has_type(x, NodeType.GROUP)):
        if not all(has_type(child, NodeType.ENT) for child in group):
            group.label = f'UNDEF_{group.oid.hex}'

    return tree


def _parse_tree_output(raw_output: str | None, *, fallback: Tree, debug: bool = False) -> Tree:
    """Try parsing raw LLM output into a Tree, fallback if parsing fails."""
    if not raw_output:
        return fallback

    try:
        json_data = json_repair.loads(raw_output, logging=debug)
        if debug:
            json_data = json_data[0]
            if json_data[1]:
                print('Fixes:', json_data[1])

        if not json_data:
            return fallback

        if isinstance(json_data, dict):
            tree = Tree.from_json(json_data)

        elif isinstance(json_data, list) and isinstance(json_data[0], dict):
            tree = Tree.from_json(json_data[0])

        else:
            return fallback

        # assign old oid to avoid duplicates
        children = [tree] if has_type(tree) else [child.detach() for child in tree]
        tree = Tree('ROOT', children, oid=fallback.oid)

        return _validate(tree)

    except ValueError as error:
        if debug:
            warnings.warn(str(error), RuntimeWarning)

    return fallback


def _build_simplify_langchain_graph(
    llm: BaseLLM,
    prompt: PromptTemplate,
    debug: bool = False,
) -> Runnable[Sequence[Tree], Sequence[Tree]]:
    to_json = RunnableLambda(lambda trees: {"trees": _tree_to_list(trees)})
    llm_chain = to_json | prompt | llm | NumberedListOutputParser()
    parallel = RunnableParallel(origin=RunnablePassthrough(), simplified=llm_chain)
    tree_parser = RunnableLambda(
        lambda result: [
            tree
            for orig, simp in itertools.zip_longest(result['origin'], result['simplified'][: len(result['origin'])])
            if (tree := _parse_tree_output(simp, fallback=orig, debug=debug))
        ]
    )

    return parallel | tree_parser


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
    chain = _build_simplify_langchain_graph(llm, prompt, debug=debug)

    def count_tokens(documents: Iterable[Tree]) -> int:
        full_prompt = prompt.format(trees=_tree_to_list(documents))
        return llm.get_num_tokens(full_prompt)

    # Group trees respecting max_token
    batches = more_itertools.constrained_batches(
        trees,
        max_size=max_token,
        get_len=count_tokens,
        strict=False,
    )

    # Run multiple group in parallel
    for batch in batches:
        yield from chain.invoke(batch)


def get_vocab(forest: Iterable[Tree], min_support: int) -> tuple[str, ...]:
    vocab_counter = Counter(
        subtree.label
        for tree in forest
        for subtree in tree.subtrees(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL}))
    )
    return tuple(label for label, cnt in vocab_counter.most_common() if cnt >= min_support)


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

    with mlflow.start_span('rewriting') if mlflow.active_run() else nullcontext():
        for iteration in trange(refining_steps + 1, leave=False, desc='rewriting iterations'):
            with (
                mlflow.start_span(
                    'iteration',
                    attributes={
                        'step': iteration,
                    },
                )
                if mlflow.active_run()
                else nullcontext()
            ):
                vocab = get_vocab(forest, min_support)

                simplification = tqdm(windowed_shuffle(forest), leave=False, total=len(forest), desc='simplifying')
                simplification = llm_simplify(llm, max_token, prompt, simplification, debug, vocab)

                if isinstance(forest, TreeBucket):
                    forest.update(simplification)

                else:
                    forest[:] = list(simplification)

                if mlflow.active_run():
                    metrics.update()
                    metrics.log_to_mlflow(iteration + 1, debug=debug)

    metrics.update()
    return metrics

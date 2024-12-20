import itertools
import json
import warnings
from collections import Counter
from collections.abc import AsyncGenerator, Collection, Iterable, Sequence
from contextlib import nullcontext

import json_repair
import mlflow
import more_itertools
from aiostream import Stream, pipe, stream
from httpx import HTTPStatusError
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from mlflow.entities import SpanEvent
from tqdm.auto import tqdm, trange

from architxt.bucket import TreeBucket
from architxt.metrics import Metrics
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC
from architxt.tree import Forest, NodeType, Tree, has_type
from architxt.utils import windowed_shuffle

__all__ = ['estimate_tokens', 'llm_rewrite']

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
You are a data-engineer agent whose task is deterministic JSON tree normalization and schema induction for noisy JSON trees.
Goal: produce one simplified, canonical JSON tree per input tree.
All tree should share the same vocabulary.
You can restructure JSON trees by adding, removing, renaming, or moving nodes.
ENT = property, GROUP = table, REL = relation.

Node format:
{{"oid":<str|null>,"name":<str>,"type":"GROUP"|"REL"|"ENT"|null,"metadata":<obj|null>,"children":[...]}}

Rules:
- Do **NOT** modify or rename ENT nodes.
- You can duplicates ENT nodes if needed.
- Return one simplified tree per input. No notes or explanations.
- Each output tree must start with root:
  {{"oid":null,"name":"ROOT","type":null,"metadata":{{}},"children":[...]}}
- Create meaningful GROUP nodes to collect related ENT nodes.
- Link GROUPs with REL nodes where appropriate.
- Preserve original oids; any new node gets "oid":null.
- Keep the tree structure as close as possible to the original one.
- Use generic semantic group names (eg. Person). Avoid dataset- or domain-specific proper nouns.
{vocab}

Your response should be a numbered list with each item on a new line (do not put linebreak in the resulting json).
For example:
1. {{...}}
2. {{...}}
3. {{...}}
"""),
        HumanMessage("""
1. {"oid":"1","name":"UNDEF","type":null,"children":[{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]},{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}]}
2. {"oid":"4","name":"UNDEF","type":null,"children":[{"oid":"5","name":"FruitName","type":"ENT","children":["orange"]},{"oid":"6","name":"PersonName","type":"ENT","children":["Alice"]},{"oid":"7","name":"Age","type":"ENT","children":["30"]}]}
        """),
        AIMessage("""
1. {"oid":null,"name":"ROOT","type":null,"children":[{"oid":"1","name":"Fruit","type":"GROUP","children":[{"oid":"2","name":"FruitName","type":"ENT","children":["banana"]},{"oid":"3","name":"Color","type":"ENT","children":["yellow"]}]}]}
2. {"oid":null,"name":"ROOT","type":null,"children":[{"oid":null,"name":"Eat","type":"REL","children":[{"oid":null,"name":"Fruit","type":"GROUP","children":[{"oid":"5","name":"FruitName","type":"ENT","children":["orange"]}]},{"oid":null,"name":"Person","type":"GROUP","children":[{"oid":"6","name":"PersonName","type":"ENT","children":["Alice"]},{"oid":"7","name":"Age","type":"ENT","children":["30"]}]}]}]}
        """),
        HumanMessagePromptTemplate.from_template("{trees}"),
    ]
)


def _trees_to_markdown_list(trees: Iterable[Tree]) -> str:
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
        raw_output = raw_output.strip()
        json_data = json_repair.loads(raw_output, logging=debug)

        if debug:
            json_data, fixes = json_data
            if fixes and (span := mlflow.get_current_active_span()):
                event = SpanEvent(name='JSON fixes', attributes=fixes)
                span.add_event(event)

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
            if span := mlflow.get_current_active_span():
                event = SpanEvent(name='JSON parsing error', attributes={'error': str(error)})
                span.add_event(event)

    return fallback


def _build_simplify_langchain_graph(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    debug: bool = False,
) -> Runnable[Sequence[Tree], Sequence[Tree]]:
    to_json = RunnableLambda(lambda trees: {"trees": _trees_to_markdown_list(trees)})
    llm_chain = (
        to_json
        | prompt
        | llm.with_retry(
            stop_after_attempt=6,
            retry_if_exception_type=(HTTPStatusError,),
        )
        | NumberedListOutputParser()
    )
    parallel = RunnableParallel(origin=RunnablePassthrough(), simplified=llm_chain)
    tree_parser = RunnableLambda(
        lambda result: [
            tree
            for orig, simp in itertools.zip_longest(result['origin'], result['simplified'][: len(result['origin'])])
            if (tree := _parse_tree_output(simp, fallback=orig, debug=debug))
        ]
    )

    return parallel | tree_parser


def count_tokens(llm: BaseLanguageModel, trees: Iterable[Tree]) -> int:
    """
    Count the number of tokens in the prompt for a set of trees.

    :param llm: LLM model to use.
    :param trees: Sequence of trees to simplify.
    :return: Number of tokens in the formatted prompt.
    """
    json_trees = _trees_to_markdown_list(trees)
    return llm.get_num_tokens(json_trees)


def estimate_tokens(
    trees: Iterable[Tree],
    llm: BaseLanguageModel,
    max_tokens: int,
    *,
    prompt: BasePromptTemplate = DEFAULT_PROMPT,
    refining_steps: int = 0,
    error_adjustment: float = 1.2,
) -> int:
    """
    Estimate the total number of tokens required for a rewrite.

    :param trees: Sequence of trees to simplify.
    :param llm: LM model to use.
    :param max_tokens: Maximum number of tokens to allow per prompt.
    :param prompt: Prompt template to use.
    :param refining_steps: Number of refining steps to perform after the initial rewrite.
    :param error_adjustment: Factor to adjust the estimated number of tokens for error.
    :return: The total number of tokens estimated for a rewrite.
    """
    prompt_tokens = llm.get_num_tokens(prompt.format())
    batches = more_itertools.constrained_batches(
        trees,
        max_size=max_tokens - prompt_tokens,
        get_len=lambda x: count_tokens(llm, [x]),
        strict=True,
    )

    tokens = sum(prompt_tokens + count_tokens(llm, batch) for batch in batches)
    return int(tokens * (refining_steps + 1) * error_adjustment)


async def llm_simplify(
    llm: BaseChatModel,
    max_tokens: int,
    prompt: ChatPromptTemplate,
    trees: Iterable[Tree],
    *,
    debug: bool,
    vocab: Collection[str] | None = None,
    task_limit: int = 4,
) -> AsyncGenerator[Tree, None]:
    """
    Simplify parse trees using an LLM.

    :param llm: LLM model to use.
    :param max_tokens: Maximum number of tokens to allow per prompt.
    :param prompt: Prompt template to use.
    :param trees: Sequence of trees to simplify.
    :param debug: Whether to enable debug logging.
    :param vocab: Optional list of vocabulary words to use in the prompt.
    :param task_limit: Maximum number of concurrent requests to make.

    :yield: Simplified trees objects with the same oid as input.
    """
    vocab_str = f"Prefer using names from this vocabulary: {', '.join(vocab)}." if vocab else ""
    prompt = prompt.partial(vocab=vocab_str)
    chain = _build_simplify_langchain_graph(llm, prompt, debug=debug)

    prompt_tokens = llm.get_num_tokens(prompt.format())

    # Group trees respecting the maximum number of tokens per prompt
    batches = more_itertools.constrained_batches(
        trees,
        max_size=max_tokens - prompt_tokens,
        get_len=lambda x: count_tokens(llm, [x]),
        strict=True,
    )

    # Run queries concurrently
    tree_stream: Stream[Sequence[Tree]] = stream.iterate(batches) | pipe.amap(
        chain.ainvoke, ordered=False, task_limit=task_limit
    )

    async with tree_stream.stream() as streamer:
        async for batch in streamer:
            for tree in batch:
                yield tree


def get_vocab(forest: Iterable[Tree], min_support: int) -> tuple[str, ...]:
    vocab_counter = Counter(
        subtree.label
        for tree in forest
        for subtree in tree.subtrees(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL}))
    )
    return tuple(label for label, cnt in vocab_counter.most_common() if cnt >= min_support)


async def llm_rewrite(
    forest: Forest,
    llm: BaseChatModel,
    max_tokens: int,
    tau: float = 0.7,
    min_support: int | None = None,
    refining_steps: int = 0,
    debug: bool = False,
    task_limit: int = 4,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    prompt: ChatPromptTemplate = DEFAULT_PROMPT,
) -> Metrics:
    """
    Rewrite a forest into a valid schema using a LLM agent.

    :param forest: A forest to be rewritten in place.
    :param llm: The LLM model to interact with for rewriting and simplification tasks.
    :param max_tokens: The token limit of the prompt.
    :param tau: Threshold for subtree similarity when clustering.
    :param min_support: Minimum support for vocab.
    :param refining_steps: Number of refining steps to perform after the initial rewrite.
    :param debug: Whether to enable debug logging.
    :param task_limit: Maximum number of concurrent requests to make.
    :param metric: The metric function used to compute similarity between subtrees.
    :param prompt: The prompt template to use for the LLM during the simplification.

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
                simplification = llm_simplify(
                    llm,
                    max_tokens,
                    prompt,
                    simplification,
                    debug=debug,
                    vocab=vocab,
                    task_limit=task_limit,
                )

                if isinstance(forest, TreeBucket):
                    await forest.async_update(simplification)

                else:
                    forest[:] = [tree async for tree in simplification]

                if mlflow.active_run():
                    metrics.update()
                    metrics.log_to_mlflow(iteration + 1, debug=debug)

    metrics.update()
    return metrics

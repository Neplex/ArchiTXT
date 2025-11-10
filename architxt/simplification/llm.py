import itertools
import json
import warnings
from collections import Counter
from collections.abc import AsyncGenerator, Collection, Iterable, Sequence
from pathlib import Path

import json_repair
import mlflow
import more_itertools
from aiostream import Stream, pipe, stream
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
from mlflow.entities import SpanEvent, SpanType
from tqdm.auto import tqdm, trange

from architxt.bucket import TreeBucket
from architxt.forest import export_forest_to_jsonl
from architxt.metrics import Metrics
from architxt.similarity import DEFAULT_METRIC, METRIC_FUNC
from architxt.tree import Forest, NodeType, Tree, TreeOID, has_type
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
    """
    Create a numbered Markdown list where each line is a JSON representation of a :py:class:`~architxt.tree.Tree`.

    :param trees: An Iterable of trees to format

    :return: A string with one line per tree in the form "N. <json>", using compact separators and stable key ordering.
    """
    return '\n'.join(
        f'{i}. {json.dumps(tree.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)}'
        for i, tree in enumerate(trees, start=1)
        if isinstance(tree, Tree)
    )


def _sanitize(tree: Tree, oid: TreeOID) -> Tree:
    """
    Sanitize a :py:class:`~architxt.tree.Tree` in-place by renaming invalid nodes with a `UNDEF_<oid>` label.

    :param tree: The tree to sanitize.
    :param oid: The Tree OID to use.

    :return: The sanitized tree.
    """
    # ensure ROOT and assign old oid to avoid duplicates
    children = [tree] if has_type(tree) else [child.detach() for child in tree]
    tree = Tree('ROOT', children, oid=oid)

    # ensure groups are valid
    for group in tree.subtrees(lambda x: has_type(x, NodeType.GROUP)):
        if not all(has_type(child, NodeType.ENT) for child in group):
            group.label = f'UNDEF_{group.oid.hex}'

    # ensure relations are valid
    for rel in tree.subtrees(lambda x: has_type(x, NodeType.REL)):
        if len(rel) != 2 or not all(has_type(child, NodeType.GROUP) for child in rel):
            rel.label = f'UNDEF_{rel.oid.hex}'

    return tree


def _parse_tree_output(raw_output: str | None, *, fallback: Tree, debug: bool = False) -> tuple[Tree, bool]:  # noqa: C901
    """
    Parse a raw LLM output string into a Tree, returning the provided fallback when parsing fails or output is empty.

    Attempts to repair and load JSON from raw_output, convert the object into a :py:class:`~architxt.tree.Tree`,
    and wrap the parsed content under a ROOT node that reuses the fallback's oid before validating the result.
    If parsing fails or the JSON does not contain a suitable object,
    the original fallback :py:class:`~architxt.tree.Tree` is returned.

    :param raw_output: The raw LLM output string to parse.
    :param fallback: The fallback original :py:class:`~architxt.tree.Tree` to return when parsing fails.
    :param debug: If True, emit warnings on parse errors and log JSON repair/parse metadata to MLflow.

    :return: The parsed :py:class:`~architxt.tree.Tree`, or the original fallback if parsing is unsuccessful.
    """
    if not raw_output:
        return fallback, False

    try:
        raw_output = raw_output.strip()
        json_data = json_repair.loads(raw_output, skip_json_loads=True, logging=debug)

        if debug:
            json_data, fixes = json_data
            if fixes and (span := mlflow.get_current_active_span()):
                event = SpanEvent(name='JSON fixes', attributes=fixes)
                span.add_event(event)

        if not json_data:
            return fallback, False

        if isinstance(json_data, dict):
            tree = Tree.from_json(json_data)

        elif isinstance(json_data, list):
            children = [Tree.from_json(sub_tree) for sub_tree in json_data if isinstance(sub_tree, dict)]
            if children:
                tree = Tree('ROOT', children)
            else:
                return fallback, False

        else:
            return fallback, False

        tree = _sanitize(tree, oid=fallback.oid)

    except ValueError as error:
        if debug:
            warnings.warn(str(error), RuntimeWarning)
            if span := mlflow.get_current_active_span():
                span.record_exception(error)

    else:
        return tree, tree != fallback

    return fallback, False


def _build_simplify_langchain_graph(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    debug: bool = False,
) -> Runnable[Sequence[Tree], Sequence[tuple[Tree, bool]]]:
    """
    Build a LangChain graph that simplifies :py:class:`~architxt.tree.Tree` using the provided model and prompt.

    :param llm: The LLM model to use for simplification.
    :param prompt: The prompt template to use for simplification.
    :param debug: If True, emit warnings on parse errors and log JSON repair/parse metadata to MLflow.

    :return: A Runnable LangChain graph that simplifies :py:class:`~architxt.tree.Tree`.
    """
    to_json = RunnableLambda(lambda trees: {"trees": _trees_to_markdown_list(trees)})
    llm_chain = to_json | prompt | llm.with_retry(stop_after_attempt=10) | NumberedListOutputParser()
    parallel = RunnableParallel(origin=RunnablePassthrough(), simplified=llm_chain)
    tree_parser = RunnableLambda(
        lambda result: tuple(
            _parse_tree_output(simplified, fallback=origin, debug=debug)
            for origin, simplified in itertools.zip_longest(
                result['origin'], result['simplified'][: len(result['origin'])]
            )
        )
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
) -> tuple[int, int, int]:
    """
    Estimate the total number of tokens (input/output) and queries required for a rewrite.

    :param trees: Sequence of trees to simplify.
    :param llm: LM model to use.
    :param max_tokens: Maximum number of tokens to allow per prompt.
    :param prompt: Prompt template to use.
    :param refining_steps: Number of refining steps to perform after the initial rewrite.
    :param error_adjustment: Factor to adjust the estimated number of tokens for error.

    :return: The total number of tokens (input/output) and the number of queries estimated for a rewrite.
    """
    prompt_tokens = llm.get_num_tokens(prompt.format(trees='', vocab=''))
    batches = more_itertools.constrained_batches(
        trees,
        max_size=max_tokens - prompt_tokens,
        get_len=lambda x: count_tokens(llm, [x]),
        strict=False,
    )

    queries = 0
    input_tokens = 0
    output_tokens = 0

    for batch in batches:
        queries += 1
        tokens = count_tokens(llm, batch)
        input_tokens += prompt_tokens + tokens
        output_tokens += tokens

    return (
        int(input_tokens * (refining_steps + 1) * error_adjustment),
        int(output_tokens * (refining_steps + 1) * error_adjustment),
        queries * (refining_steps + 1),
    )


async def llm_simplify(
    llm: BaseChatModel,
    max_tokens: int,
    prompt: ChatPromptTemplate,
    trees: Iterable[Tree],
    *,
    debug: bool,
    vocab: Collection[str] | None = None,
    task_limit: int = 4,
) -> AsyncGenerator[tuple[Tree, bool], None]:
    """
    Simplify parse trees using an LLM.

    It uses the following flow where the tree parser falls back to the original tree in case of parsing errors:

    .. mermaid::
        :alt: ArchiTXT Schema
        :align: center

        ---
        config:
          theme: neutral
        ---
        flowchart LR
            A[Trees] --> B[Convert to JSON] --> C[LLM]
            A & C --> E[Tree parser]
            E --> F[Simplified trees]

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

    prompt_tokens = llm.get_num_tokens(prompt.format(trees=''))

    # Group trees respecting the maximum number of tokens per prompt
    batches = more_itertools.constrained_batches(
        trees,
        max_size=max_tokens - prompt_tokens,
        get_len=lambda x: count_tokens(llm, [x]),
        strict=False,
    )

    @mlflow.trace(name='llm-invoke', span_type=SpanType.CHAIN)
    async def _safe_traced_invoke(tree_batch: Sequence[Tree]) -> Sequence[tuple[Tree, bool]]:
        try:
            return await chain.ainvoke(tree_batch)

        except Exception as error:
            warnings.warn(str(error), RuntimeWarning)
            if span := mlflow.get_current_active_span():
                span.record_exception(error)

            return [(orig_tree, False) for orig_tree in tree_batch]

    # Run queries concurrently
    tree_stream: Stream[Sequence[tuple[Tree, bool]]] = stream.iterate(batches) | pipe.amap(
        _safe_traced_invoke, ordered=False, task_limit=task_limit
    )

    async with tree_stream.stream() as streamer:
        async for batch in streamer:
            for tree, simplified in batch:
                yield tree, simplified


@mlflow.trace(span_type=SpanType.PARSER)
def get_vocab(forest: Iterable[Tree], min_support: int) -> set[str]:
    """
    Extract a set of labels that appear in GROUP or REL subtrees with at least a given support.

    :param forest: Forest to extract vocabulary from.
    :param min_support: Minimum support threshold for vocabulary.
    :return: Set of labels.
    """
    vocab_counter = Counter(
        subtree.label
        for tree in forest
        for subtree in tree.subtrees(lambda x: has_type(x, {NodeType.GROUP, NodeType.REL}))
    )
    return {label for label, cnt in vocab_counter.most_common() if cnt >= min_support}


async def llm_rewrite(
    forest: Forest,
    llm: BaseChatModel,
    max_tokens: int,
    tau: float = 0.7,
    min_support: int | None = None,
    refining_steps: int = 0,
    debug: bool = False,
    intermediate_output_path: Path | None = None,
    task_limit: int = 1,
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
    :param intermediate_output_path: Optional path to save intermediate results after each iteration.
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

    with mlflow.start_span('llm-rewriting', span_type=SpanType.CHAIN):
        for iteration in trange(refining_steps + 1, leave=False, desc='rewriting iterations'):
            with mlflow.start_span(
                'llm-rewriting-iteration',
                span_type=SpanType.CHAIN,
                attributes={
                    'step': iteration,
                },
            ) as iteration_span:
                vocab = get_vocab(forest, min_support)
                iteration_span.set_attribute('vocab', sorted(vocab))

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

                # Track if any tree was modified
                any_modified = False

                async def _simplification_wrap() -> AsyncGenerator[Tree, None]:
                    nonlocal any_modified
                    async for tree, simplified in simplification:
                        if simplified:
                            any_modified = True
                        yield tree

                if isinstance(forest, TreeBucket):
                    await forest.async_update(_simplification_wrap())
                else:
                    forest[:] = [tree async for tree in _simplification_wrap()]

                iteration_span.set_attribute('simplified', any_modified)

                # Save intermediate results
                if intermediate_output_path:
                    intermediate_output_path.mkdir(parents=True, exist_ok=True)
                    intermediate_file = intermediate_output_path / f'intermediate_{iteration}.jsonl'
                    export_forest_to_jsonl(intermediate_file, forest)

                # Log metrics to MLflow
                if mlflow.active_run():
                    metrics.update()
                    metrics.log_to_mlflow(iteration + 1, debug=debug)

                # Early stopping if no tree was modified
                if not any_modified:
                    break

    metrics.update()
    return metrics

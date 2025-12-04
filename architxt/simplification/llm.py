from __future__ import annotations

import itertools
import json
import re
import unicodedata
import warnings
from collections import Counter
from difflib import get_close_matches
from typing import TYPE_CHECKING

import json_repair
import mlflow
import more_itertools
from aiostream import Stream, pipe, stream
from json_repair import JSONReturnType
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
from architxt.schema import Schema
from architxt.similarity import DECAY, DEFAULT_METRIC, METRIC_FUNC
from architxt.tree import Forest, NodeLabel, NodeType, Tree, TreeOID, has_type
from architxt.utils import windowed_shuffle

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Collection, Iterable, Sequence
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel, BaseLanguageModel

__all__ = ['estimate_tokens', 'llm_rewrite']

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
You are a data-engineer agent whose task is deterministic JSON tree normalization and schema induction for noisy JSON trees.
Goal: produce one simplified, canonical JSON tree per input tree.
You can restructure JSON trees by adding, removing, renaming, or moving nodes.
ENT = property, GROUP = table, REL = relation.
Keep existing groups and relations when possible.
All trees should share the same vocabulary, rename groups and relations according to the relevant vocabulary.
{vocab}

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
- Create generic semantic group names (eg. Person). Avoid dataset- or domain-specific names (eg. prefer Exam over EGC).

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
    return '\n\n'.join(
        f'{i}. {json.dumps(tree.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)}'
        for i, tree in enumerate(trees, start=1)
        if isinstance(tree, Tree)
    )


def _parse_tree(json_data: JSONReturnType) -> Tree:
    """
    Parse a JSON object into a Tree.

    :param json_data: The JSON object to parse.
    :raise ValueError: If the JSON object is not a valid tree.
    :raise TypeError: If the JSON object is of an invalid type.
    :return: The parsed Tree.
    """
    if not json_data:
        msg = 'Empty JSON data cannot be parsed into a tree.'
        raise TypeError(msg)

    if isinstance(json_data, dict):
        tree = Tree.from_json(json_data)

    elif isinstance(json_data, list):
        children = [Tree.from_json(sub_tree) for sub_tree in json_data if isinstance(sub_tree, dict)]
        if children:
            tree = Tree('ROOT', children)
        else:
            msg = 'No valid tree objects found in JSON list data.'
            raise ValueError(msg)

    else:
        msg = f'Invalid JSON data type for tree parsing: {type(json_data)}.'
        raise TypeError(msg)

    return tree


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

    # ensure groups and relations are valid
    for st in tree.subtrees(reverse=True):
        if (has_type(st, NodeType.GROUP) and not all(has_type(c, NodeType.ENT) for c in st)) or (
            has_type(st, NodeType.REL) and (len(st) != 2 or not all(has_type(c, NodeType.GROUP) for c in st))
        ):
            st.label = f'UNDEF_{st.oid.hex}'

    return tree


def _fix_vocab(tree: Tree, vocab: Collection[str], vocab_similarity: float = 0.6) -> Tree:
    """
    Fix the vocabulary in the tree by updating GROUP and REL labels in-place to match canonical forms.

    :param tree: Trees to fix.
    :param vocab: Collection of canonical labels.
    :param vocab_similarity: Similarity threshold in [0, 1] for merging labels.
    :return: An updated tree with fixed vocabulary.
    """
    for subtree in tree.subtrees():
        if (
            has_type(subtree, {NodeType.GROUP, NodeType.REL})
            and (label := _normalize(subtree.label.name))
            and (matches := get_close_matches(label, vocab, n=1, cutoff=vocab_similarity))
        ):
            subtree.label = NodeLabel(subtree.label.type, matches[0])

    return tree


def _parse_tree_output(
    raw_output: str | None,
    *,
    fallback: Tree,
    vocab: Collection[str] | None = None,
    vocab_similarity: float = 0.6,
    debug: bool = False,
) -> tuple[Tree, bool]:
    """
    Parse a raw LLM output string into a Tree, returning the provided fallback when parsing fails or output is empty.

    Attempts to repair and load JSON from raw_output, convert the object into a :py:class:`~architxt.tree.Tree`,
    and wrap the parsed content under a ROOT node that reuses the fallback's oid before validating the result.
    If parsing fails or the JSON does not contain a suitable object,
    the original fallback :py:class:`~architxt.tree.Tree` is returned.

    :param raw_output: The raw LLM output string to parse.
    :param fallback: The fallback original :py:class:`~architxt.tree.Tree` to return when parsing fails.
    :param vocab: Collection of canonical labels.
    :param vocab_similarity: Similarity threshold in [0, 1] for merging labels.
    :param debug: If True, emit warnings on parse errors and log JSON repair/parse metadata to MLflow.

    :return: The parsed :py:class:`~architxt.tree.Tree`, or the original fallback if parsing is unsuccessful.
    """
    if not raw_output:
        return fallback, False

    try:
        raw_output = raw_output.strip()
        json_data = json_repair.loads(raw_output, skip_json_loads=True, logging=debug)

        if isinstance(json_data, tuple):
            json_data, fixes = json_data
            if fixes and (span := mlflow.get_current_active_span()):
                event = SpanEvent(
                    name='JSON fixes', attributes={'json_fixes': [fix['text'] for fix in fixes if 'text' in fix]}
                )
                span.add_event(event)

        tree = _parse_tree(json_data)
        tree = _sanitize(tree, oid=fallback.oid)

        if vocab:
            tree = _fix_vocab(tree, vocab=vocab, vocab_similarity=vocab_similarity)

    except (ValueError, TypeError) as error:
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
    *,
    vocab: Collection[str] | None = None,
    vocab_similarity: float = 0.6,
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
            _parse_tree_output(simplified, fallback=origin, vocab=vocab, vocab_similarity=vocab_similarity, debug=debug)
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
    vocab: Collection[str] | None = None,
    vocab_similarity: float = 0.6,
    task_limit: int = 4,
    debug: bool = False,
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
    :param vocab: Optional list of vocabulary words to use in the prompt.
    :param vocab_similarity: Similarity threshold in [0, 1] for merging labels.
    :param task_limit: Maximum number of concurrent requests to make.
    :param debug: Whether to enable debug logging.

    :yield: Simplified trees objects with the same oid as input.
    """
    vocab_str = f"Prefer these labels : {', '.join(vocab)}." if vocab else ""
    prompt = prompt.partial(vocab=vocab_str)
    chain = _build_simplify_langchain_graph(llm, prompt, vocab=vocab, vocab_similarity=vocab_similarity, debug=debug)

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


def _normalize(s: str) -> str:
    """
    Normalize a string for vocabulary extraction.

    Applies Unicode NFKC normalization, removes non-alphanumeric characters,
    and converts to upper snake_case (e.g., "hello, world" -> "HELLO_WORLD").

    :param s: String to normalize.
    :return: Normalized upper snake_case string, or empty string if no alphanumeric characters.
    """
    # unicode normalize
    s = unicodedata.normalize('NFKC', s)
    # keep alnum and spaces
    s = ''.join(ch if ch.isalnum() else ' ' for ch in s)
    # convert to upper case
    s = s.strip().upper()
    # convert to snake_case
    return re.sub(r'\s+', '_', s)


@mlflow.trace(span_type=SpanType.PARSER)
def extract_vocab(forest: Forest, min_support: int, min_similarity: float, close_match: int = 3) -> set[str]:
    """
    Extract a normalized set of labels that appear in GROUP or REL subtrees with at least a given support.

    - Normalization: Unicode NFKC, remove non-alphanumeric chars, collapse spaces, upper snake_case.
    - Aggregation: merge labels that are similar above `min_similarity` (SequenceMatcher ratio).
        We select the one with the most occurrences as canonical label if multiple match.
    - Returns: Set of canonical labels.

    :param forest: Forest to extract vocabulary from.
    :param min_support: Minimum support threshold for vocabulary.
    :param min_similarity: Similarity threshold in [0, 1] for merging labels.
    :param close_match: Number of close matches to consider when merging labels.
    :return: Set of canonical labels.
    """
    vocab_counter: Counter[str] = Counter()

    for tree in forest:
        for subtree in tree.subtrees():
            if not has_type(subtree, {NodeType.GROUP, NodeType.REL}):
                continue

            if not subtree.label or not (label := _normalize(subtree.label.name)):
                continue

            matches = get_close_matches(label, vocab_counter.keys(), n=close_match, cutoff=min_similarity)
            canonical_label = max(matches, default=label, key=lambda x: vocab_counter[x])
            vocab_counter.update([canonical_label])

    selected_vocab = {label for label, cnt in vocab_counter.items() if cnt >= min_support}

    if span := mlflow.get_current_active_span():
        span.set_attributes(
            {
                'vocab.total_unique': len(vocab_counter),
                'vocab.selected_size': len(selected_vocab),
                'vocab.top10_counts': sorted(vocab_counter.most_common(10), key=lambda x: x[1], reverse=True),
            }
        )

    return selected_vocab


def _get_mlflow_schema(forest: Forest) -> dict:
    schema = Schema.from_forest(forest)
    return {
        'forest.size': len(forest),
        'schema.size': len(schema.productions()),
        'schema.entities': sorted(schema.entities),
        'schema.groups': sorted({group.name for group in schema.groups}),
        'schema.relations': sorted({relation.name for relation in schema.relations}),
    }


async def llm_rewrite(
    forest: Forest,
    llm: BaseChatModel,
    max_tokens: int,
    tau: float = 0.7,
    decay: float = DECAY,
    min_support: int | None = None,
    vocab_similarity: float = 0.6,
    refining_steps: int = 0,
    debug: bool = False,
    intermediate_output_path: Path | None = None,
    task_limit: int = 1,
    metric: METRIC_FUNC = DEFAULT_METRIC,
    prompt: ChatPromptTemplate = DEFAULT_PROMPT,
    commit: bool | int = True,
) -> Metrics:
    """
    Rewrite a forest into a valid schema using a LLM agent.

    :param forest: A forest to be rewritten in place.
    :param llm: The LLM model to interact with for rewriting and simplification tasks.
    :param max_tokens: The token limit of the prompt.
    :param tau: Threshold for subtree similarity when clustering.
    :param decay: The similarity decay factor.
        The higher the value, the more the weight of context decreases with distance.
    :param min_support: Minimum support for vocab.
    :param vocab_similarity: Similarity threshold in [0, 1] for merging vocabulary labels.
    :param refining_steps: Number of refining steps to perform after the initial rewrite.
    :param debug: Whether to enable debug logging.
    :param intermediate_output_path: Optional path to save intermediate results after each iteration.
    :param task_limit: Maximum number of concurrent requests to make.
    :param metric: The metric function used to compute similarity between subtrees.
    :param prompt: The prompt template to use for the LLM during the simplification.
    :param commit: Commit automatically if using TreeBucket. If already in a transaction, no commit is applied.
        - If False, no commits are made, it relies on the current transaction.
        - If True (default), commits in batch.
        - If an integer, commits every N tree.
        To avoid memory issues, we recommend using incremental commit with large iterables.

    :return: A `Metrics` object encapsulating the results and metrics calculated for the LLM rewrite process.
    """
    metrics = Metrics(forest, tau=tau, decay=decay, metric=metric)
    min_support = min_support or max((len(forest) // 20), 2)

    if mlflow.active_run():
        mlflow.log_params(
            {
                'nb_sentences': len(forest),
                'tau': tau,
                'decay': decay,
                'min_support': min_support,
                'vocab_similarity': vocab_similarity,
                'metric': metric.__name__,
                'refining_steps': refining_steps,
            }
        )
        metrics.log_to_mlflow(0, debug=debug)

    mlflow_schema = _get_mlflow_schema(forest)

    for iteration in trange(refining_steps + 1, leave=False, desc='rewriting iterations'):
        with mlflow.start_span(
            'llm-rewriting',
            span_type=SpanType.CHAIN,
            attributes={
                'step': iteration,
            },
        ) as iteration_span:
            iteration_span.set_inputs(mlflow_schema)

            vocab = extract_vocab(forest, min_support, vocab_similarity)

            shuffled_forest = tqdm(windowed_shuffle(forest), leave=False, total=len(forest), desc='simplifying')
            simplification = llm_simplify(
                llm,
                max_tokens,
                prompt,
                shuffled_forest,
                vocab=vocab,
                vocab_similarity=vocab_similarity,
                task_limit=task_limit,
                debug=debug,
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
                await forest.async_update(_simplification_wrap(), commit=commit)
            else:
                forest[:] = [tree async for tree in _simplification_wrap()]

            mlflow_schema = _get_mlflow_schema(forest)
            iteration_span.set_outputs(mlflow_schema)
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

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
    """
    Create a numbered Markdown list where each line is a compact, stable JSON representation of a Tree.
    
    Parameters:
        trees (Iterable[Tree]): Iterable of items; only elements that are instances of Tree are included.
    
    Returns:
        str: A string with one line per tree in the form "N. <json>", using compact separators, stable key ordering, and UTF-8 characters preserved.
    """
    return '\n'.join(
        f'{i}. {json.dumps(tree.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)}'
        for i, tree in enumerate(trees, start=1)
        if isinstance(tree, Tree)
    )


def _validate(tree: Tree) -> Tree:
    """
    Validate and sanitize a Tree by marking invalid REL and GROUP nodes with an `UNDEF_<oid>` label.
    
    For each REL node: if it does not have exactly two children or any child is not a GROUP, its label is replaced with `UNDEF_<oid>` where `<oid>` is the node's hex oid.
    For each GROUP node: if any child is not an ENT, its label is replaced with `UNDEF_<oid>`.
    
    Returns:
        tree (Tree): The original tree object, possibly modified in place with updated labels.
    """
    for rel in tree.subtrees(lambda x: has_type(x, NodeType.REL)):
        if len(rel) != 2 or not all(has_type(child, NodeType.GROUP) for child in rel):
            rel.label = f'UNDEF_{rel.oid.hex}'

    for group in tree.subtrees(lambda x: has_type(x, NodeType.GROUP)):
        if not all(has_type(child, NodeType.ENT) for child in group):
            group.label = f'UNDEF_{group.oid.hex}'

    return tree


def _parse_tree_output(raw_output: str | None, *, fallback: Tree, debug: bool = False) -> Tree:
    """
    Parse a raw LLM output string into a Tree, returning the provided fallback when parsing fails or output is empty.
    
    Attempts to repair and load JSON from raw_output, convert the first object (or the object itself) into a Tree, and wrap the parsed content under a ROOT node that reuses the fallback's oid before validating the result. If parsing fails or the JSON does not contain a suitable object, the original fallback Tree is returned.
    
    Parameters:
        raw_output (str | None): Raw text produced by an LLM; may be None or empty.
        fallback (Tree): Tree to return when parsing fails; its oid is reused for the resulting ROOT node when parsing succeeds.
        debug (bool): If True, emit warnings on parse errors and attach JSON repair/parse metadata to an active MLflow span when available.
    
    Returns:
        Tree: The parsed and validated Tree wrapped under a ROOT node that reuses fallback.oid, or the original fallback if parsing is unsuccessful.
    """
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
    """
    Builds a LangChain Runnable graph that simplifies Tree objects using the provided chat model and prompt.
    
    Parameters:
        llm (BaseChatModel): Chat model used to generate simplified tree outputs.
        prompt (ChatPromptTemplate): Prompt template that guides the model's rewrite of the trees.
        debug (bool): If true, enables more verbose parsing behavior for LLM outputs.
    
    Returns:
        Runnable[Sequence[Tree], Sequence[Tree]]: A runnable that accepts a sequence of Trees and returns a sequence of simplified Trees in the same order; when an output cannot be parsed, the original tree is used as the fallback.
    """
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
    Compute the number of tokens required to represent the given trees when formatted for the model.
    
    Returns:
        Number of tokens required to represent the given trees when formatted for the model.
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
    Estimate total tokens required to rewrite a collection of trees with the given model and prompt.
    
    This accounts for prompt tokens, batches the trees to respect the per-prompt max_tokens limit, sums tokens for each batch, and scales the total by (refining_steps + 1) and the error_adjustment factor.
    
    Parameters:
        refining_steps (int): Number of additional refine passes to include beyond the initial rewrite.
        error_adjustment (float): Multiplicative safety factor applied to the final estimate.
    
    Returns:
        int: Estimated total number of tokens required.
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
    Produce simplified Tree objects from the input iterable using the provided chat model and prompt.
    
    Parameters:
        llm (BaseChatModel): Chat model used to generate simplifications.
        max_tokens (int): Maximum number of tokens allowed for each prompt (including prompt overhead).
        prompt (ChatPromptTemplate): Prompt template; a partial with `vocab` is applied when `vocab` is provided.
        trees (Iterable[Tree]): Trees to simplify.
        debug (bool): Enable additional debug logging and diagnostic metadata when available.
        vocab (Collection[str] | None): Optional vocabulary terms the model should prefer when normalizing labels.
        task_limit (int): Maximum number of concurrent LLM requests.
    
    Returns:
        Tree: Simplified Tree objects preserving their original `oid`, yielded as results become available.
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
    """
    Build a vocabulary of labels that appear in GROUP or REL subtrees with at least a given support.
    
    Parameters:
        forest (Iterable[Tree]): An iterable of Tree objects to extract labels from.
        min_support (int): Minimum number of occurrences required for a label to be included.
    
    Returns:
        tuple[str, ...]: Labels (in descending frequency order) whose occurrence count is greater than or equal to `min_support`.
    """
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
    Rewrite the given forest in place using an LLM-driven simplification pipeline and return run metrics.
    
    Parameters:
    	forest (Forest): Forest to be rewritten; its contents are updated in place.
    	llm (BaseChatModel): Chat model used to perform simplification and normalization.
    	max_tokens (int): Maximum token budget per LLM request (used to batch trees).
    	tau (float): Similarity threshold used by the metrics/clusterer when assessing subtree similarity.
    	min_support (int | None): Minimum occurrence count for labels to be included in the vocabulary; if None a default is derived from forest size.
    	refining_steps (int): Number of additional refinement iterations to run after the initial rewrite (0 runs only the initial pass).
    	debug (bool): Enable verbose/debugging behaviors where supported.
    	task_limit (int): Maximum number of concurrent LLM tasks to run.
    	metric (METRIC_FUNC): Function used to compute subtree similarity for metrics and clustering.
    	prompt (ChatPromptTemplate): Prompt template driving the LLM simplification instructions.
    
    Returns:
    	Metrics: Object containing computed metrics and summary information for the rewriting process.
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
from functools import partial

import anyio
import streamlit as st

from architxt.bucket import TreeBucket
from architxt.llm import get_chat_model
from architxt.simplification.llm import llm_rewrite
from architxt.simplification.tree_rewriting import rewrite
from architxt.ui.utils import get_forest, update_metrics


def _render_rule_based_simplification(forest: TreeBucket) -> None:
    c1, c2 = st.columns(2)
    tau = c1.number_input('Tau (Similarity)', 0.0, 1.0, 0.8)
    decay = c2.number_input('Decay', min_value=0.0, value=2.0)
    min_support = c1.number_input('Min Support', min_value=1, value=10)
    epoch = c2.number_input('Epochs', min_value=1, value=50)
    schema_similarity = st.selectbox("Compute similarity on", ('instance', 'schema'), index=0)

    if st.button("Apply Rule-Based Simplification"):
        with st.spinner("Simplifying..."):
            rewrite(
                forest,
                tau=tau,
                decay=decay,
                epoch=epoch,
                min_support=min_support,
                schema_similarity=schema_similarity == 'schema',
            )

        update_metrics()


def _render_llm_based_simplification(forest: TreeBucket) -> None:
    st.warning("LLM Simplification requires extra dependencies and an LLM provider.")

    local: bool = st.session_state.get("llm_local", False)
    c1, c2 = st.columns(2)
    model_provider = c1.text_input("Provider", "huggingface", disabled=local)
    model_name = c2.text_input("Model", value="deepseek-ai/DeepSeek-V3")

    with st.expander("Advanced LLM Settings"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.number_input("Temperature", 0.0, 1.0, 0.2)
        max_tokens = c2.number_input("Max Tokens", min_value=256, value=4096, step=128)
        refining_steps = c3.number_input("Refining Steps", min_value=0, value=3)

        tau = c1.number_input('Tau (Similarity)', 0.0, 1.0, 0.8)
        decay = c2.number_input('Decay', min_value=0.0, value=2.0)
        min_support = c3.number_input('Min Support', min_value=1, value=10)

    if st.button("Apply LLM Simplification"):
        with st.status("Simplifying..."):
            st.write("Loading LLM...")
            llm = get_chat_model(
                model_provider,
                model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                local=local,
                openvino=st.session_state.get("llm_openvino", False),
            )
            transform = partial(
                llm_rewrite,
                llm=llm,
                max_tokens=max_tokens,
                tau=tau,
                decay=decay,
                min_support=min_support,
                refining_steps=refining_steps,
            )
            st.write("Simplifying...")
            anyio.run(transform, forest)

        update_metrics()


TRANSFORMER = {
    "Default (rule-based)": _render_rule_based_simplification,
    "LLM": _render_llm_based_simplification,
}


@st.fragment
def simplification() -> None:
    st.header("Simplify")

    forest = get_forest()

    if len(forest) == 0:
        st.info("No data loaded. Please import data using the top menu.")

    else:
        method = st.selectbox("Method", TRANSFORMER.keys())
        TRANSFORMER[method](forest)

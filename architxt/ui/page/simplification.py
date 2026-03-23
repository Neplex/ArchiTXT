from functools import partial
from typing import TYPE_CHECKING

import anyio
import streamlit as st

from architxt.bucket import TreeBucket
from architxt.simplification.llm import llm_rewrite
from architxt.simplification.tree_rewriting import rewrite
from architxt.ui.utils import get_forest, update_metrics

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def _render_rule_based_simplification(forest: TreeBucket) -> None:
    c1, c2 = st.columns(2)
    tau = c1.number_input('Tau (Similarity)', 0.0, 1.0, 0.8)
    decay = c2.number_input('Decay', min_value=0.0, value=2.0)
    min_support = c1.number_input('Min Support', min_value=1, value=10)
    epoch = c2.number_input('Epochs', min_value=1, value=50)

    if st.button("Apply Rule-Based Simplification"):
        with st.spinner("Simplifying..."):
            rewrite(forest, tau=tau, decay=decay, epoch=epoch, min_support=min_support)

        update_metrics()


@st.cache_resource(scope="session")
def get_llm(
    provider: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    openvino: bool = False,
) -> 'BaseChatModel':
    from langchain.chat_models import init_chat_model

    if provider == "Local (HuggingFace)":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task='text-generation',
            device_map=None if openvino else 'auto',
            backend='openvino' if openvino else 'pt',
            model_kwargs={'export': True} if openvino else {'torch_dtype': 'auto'},
            pipeline_kwargs={
                'use_cache': True,
                'do_sample': True,
                'return_full_text': False,
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'repetition_penalty': 1.1,
                'num_return_sequences': 1,
                'pad_token_id': 0,
            },
        )
        return ChatHuggingFace(llm=pipeline)

    return init_chat_model(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _render_llm_based_simplification(forest: TreeBucket) -> None:
    st.warning("LLM Simplification requires extra dependencies and an LLM provider.")

    openvino = False
    provider = st.selectbox("Provider", ["Local (HuggingFace)", "API"])
    model_name = st.text_input(
        "Model Name", value="deepseek-ai/DeepSeek-V3" if provider == "Local (HuggingFace)" else "openai:o3-mini"
    )

    with st.expander("Advanced LLM Settings"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.number_input("Temperature", 0.0, 1.0, 0.2)
        max_tokens = c2.number_input("Max Tokens", min_value=256, value=4096, step=128)
        refining_steps = c3.number_input("Refining Steps", min_value=0, value=3)

        tau = c1.number_input('Tau (Similarity)', 0.0, 1.0, 0.8)
        decay = c2.number_input('Decay', min_value=0.0, value=2.0)
        min_support = c3.number_input('Min Support', min_value=1, value=10)

        if provider == "Local (HuggingFace)":
            openvino = st.checkbox("Use OpenVINO (if available)", value=False)

    if st.button("Apply LLM Simplification"):
        llm = get_llm(provider, model_name, max_tokens, temperature, openvino)
        transform = partial(
            llm_rewrite,
            llm=llm,
            max_tokens=max_tokens,
            tau=tau,
            decay=decay,
            min_support=min_support,
            refining_steps=refining_steps,
        )

        with st.spinner("Simplifying with LLM..."):
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

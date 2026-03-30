import pandas as pd
import streamlit as st

from architxt.bucket import TreeBucket
from architxt.labelling import Renaming, apply_renaming, llm_group_labelling, llm_relation_labelling
from architxt.llm import get_chat_model
from architxt.schema import Schema
from architxt.tree import NodeType
from architxt.ui.utils import get_forest, get_schema, update_metrics


def _render_editor(forest: TreeBucket, schema: Schema) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Groups**")
        st.session_state.group_renames = st.data_editor(
            st.session_state.group_renames,
            key="group_editor",
            hide_index=True,
            width="stretch",
            disabled=["Current Name"],
        )

    with col2:
        st.write("**Relations**")
        st.session_state.relation_renames = st.data_editor(
            st.session_state.relation_renames,
            key="relation_editor",
            hide_index=True,
            width="stretch",
            disabled=["Current Name"],
        )

    renames = []
    for _, row in st.session_state.group_renames.iterrows():
        if row["Current Name"] != row["New Name"]:
            renames.append(Renaming(NodeType.GROUP, row["Current Name"], row["New Name"]))

    for _, row in st.session_state.relation_renames.iterrows():
        if row["Current Name"] != row["New Name"]:
            renames.append(Renaming(NodeType.REL, row["Current Name"], row["New Name"]))

    c1, c2 = st.columns(2)
    if c1.button("Reset", width="stretch"):
        _reset_labelling_tables(schema)
        st.rerun()

    if c2.button("Apply Renaming", width="stretch", disabled=not renames):
        with (
            st.spinner("Applying new labels to forest..."),
            forest.transaction(),
        ):
            apply_renaming(forest, renames)

        st.toast(f"Applied {len(renames)} renaming.")
        update_metrics()


def _render_llm_labelling(forest: TreeBucket, schema: Schema) -> None:
    st.subheader('LLM Auto-labelling')

    st.warning("LLM Labelling requires extra dependencies and an LLM provider.")

    local: bool = st.session_state.get("llm_local", False)
    c1, c2 = st.columns(2)
    model_provider = c1.text_input("Provider", "huggingface", disabled=local)
    model_name = c2.text_input("Model", value="deepseek-ai/DeepSeek-V3")

    with st.expander("Advanced LLM Settings"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.number_input("Temperature", 0.0, 1.0, 0.2)
        max_tokens = c2.number_input("Max Tokens", min_value=256, value=4096, step=128)
        sample_size = c3.number_input("Sample Size", min_value=0, value=5)

    if st.button("Get AI Suggestions"):
        with st.status("Computing new labels..."):
            st.write("Loading LLM...")
            llm = get_chat_model(
                model_provider,
                model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                local=local,
                openvino=st.session_state.get("llm_openvino", False),
            )

            st.write("Generating group names...")
            groups_renames = llm_group_labelling(schema, llm, forest=forest, sample_size=sample_size)

            # Update group dataframe in session state
            new_group_df = st.session_state.group_renames.copy()
            rename_dict = {r.old_name: r.new_name for r in groups_renames}
            new_group_df["New Name"] = new_group_df["Current Name"].map(lambda x: rename_dict.get(x, x))
            st.session_state.group_renames = new_group_df

            st.write("Generating relation names...")
            relations_renames = llm_relation_labelling(schema, llm, group_renames=groups_renames)

            # Update relation dataframe in session state
            new_relation_df = st.session_state.relation_renames.copy()
            rel_rename_dict = {r.old_name: r.new_name for r in relations_renames}
            new_relation_df["New Name"] = new_relation_df["Current Name"].map(lambda x: rel_rename_dict.get(x, x))
            st.session_state.relation_renames = new_relation_df

            st.toast("AI suggestions generated! Review them in the tables above.")
            st.rerun()


def _reset_labelling_tables(schema: Schema) -> None:
    group_names = {g.name for g in schema.groups}
    st.session_state.group_renames = pd.DataFrame([{"Current Name": g, "New Name": g} for g in sorted(group_names)])

    rels_names = {r.name for r in schema.relations}
    st.session_state.relation_renames = pd.DataFrame([{"Current Name": r, "New Name": r} for r in sorted(rels_names)])


@st.fragment
def labelling() -> None:
    st.header("Labelling")

    forest = get_forest()
    schema = get_schema()

    if len(schema.groups) == 0:
        st.warning("No groups found in the forest.")
        return

    # Initialize session state for renaming
    if "group_renames" not in st.session_state:
        _reset_labelling_tables(schema)

    _render_editor(forest, schema)
    st.divider()
    _render_llm_labelling(forest, schema)

import mlflow
import streamlit as st

from architxt.ui.page import exporter, importer, labelling, simplification, visualizer
from architxt.ui.utils import clear_data, get_metrics

PAGES = {
    "File": [
        st.Page(importer, title="Import", icon=":material/download:"),
        st.Page(exporter, title="Export", icon=":material/upload:"),
    ],
    "Tools": [
        st.Page(visualizer, title="Visualize", icon=":material/table:"),
        st.Page(labelling, title="Labelling", icon=":material/ink_pen:"),
        st.Page(simplification, title="Simplify", icon=":material/build:"),
    ],
}


@st.fragment()
def render_metrics() -> None:
    prev, curr = get_metrics()

    cols = st.columns(len(curr))
    for col, label in zip(cols, curr.keys()):
        col.metric(label, curr[label], delta=(curr[label] - prev[label]) or None)


def main() -> None:
    st.set_page_config(page_title="ArchiTXT UI", layout="wide")
    st.title("ArchiTXT")

    pg = st.navigation(PAGES, position="top")

    # Sidebar - Global Settings
    with st.sidebar:
        st.header("Settings")
        st.session_state["cache"] = st.toggle("Enable Cache", key="nlp_cache", value=True)
        st.session_state["mlflow_enabled"] = st.toggle("Enable MLFlow Logging", key="mlflow_toggle")

        if st.session_state["mlflow_enabled"]:
            mlflow.set_experiment('ArchiTXT UI')
            st.success(f"Logging to {mlflow.get_tracking_uri()}")

        st.session_state["llm_local"] = st.toggle("Local LLM", key="w_llm_local", value=False)
        st.session_state["llm_openvino"] = st.toggle("Use OpenVINO", key="w_llm_openvino", value=False)

    render_metrics()

    if st.button("Clear data"):
        clear_data()

    pg.run()


if __name__ == "__main__":
    main()

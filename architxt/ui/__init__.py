import mlflow
import streamlit as st

from architxt.schema import Schema
from architxt.ui.page import exporter, importer, simplification, visualizer
from architxt.ui.utils import get_forest


@st.fragment
def render_metrics() -> None:
    forest = get_forest()
    forest.sync()

    schema = Schema.from_forest(forest)

    metrics = {
        "Total Trees": len(forest),
        "Entities": len(schema.entities),
        "Groups": len(schema.groups),
        "Relations": len(schema.relations),
    }

    prev = st.session_state.get("prev_metrics")
    if prev is None:
        prev = metrics.copy()

    cols = st.columns(len(metrics))
    for col, label in zip(cols, metrics.keys()):
        col.metric(label, metrics[label], delta=(metrics[label] - prev[label]) or None)

    st.session_state["prev_metrics"] = metrics

    if st.button("Clear data"):
        with forest.transaction():
            forest.clear()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ArchiTXT UI", layout="wide")
    st.title("ArchiTXT")

    # Sidebar - Global Settings
    with st.sidebar:
        st.header("Settings")
        st.session_state["cache"] = st.toggle("Enable Cache", key="nlp_cache", value=True)
        st.session_state["mlflow_enabled"] = st.toggle("Enable MLFlow Logging", key="mlflow_toggle")

        if st.session_state["mlflow_enabled"]:
            mlflow.set_experiment('ArchiTXT UI')
            st.success(f"Logging to {mlflow.get_tracking_uri()}")

    render_metrics()

    pages = {
        "File": [
            st.Page(importer, title="Import", icon=":material/download:"),
            st.Page(exporter, title="Export", icon=":material/upload:"),
        ],
        "Tools": [
            st.Page(visualizer, title="Visualize", icon=":material/table:"),
            st.Page(simplification, title="Simplify", icon=":material/build:"),
        ],
    }

    pg = st.navigation(pages, position="top")
    pg.run()


if __name__ == "__main__":
    main()

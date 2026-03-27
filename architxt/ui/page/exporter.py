import tempfile
from pathlib import Path

import streamlit as st

from architxt.bucket import TreeBucket
from architxt.database import export
from architxt.forest import export_forest_to_jsonl
from architxt.ui.utils import get_forest, get_neo4j_driver, get_sql_engine


def _render_sql_export(forest: TreeBucket) -> None:
    sql_uri = st.text_input("SQL URI", value="sqlite:///output.db", help="e.g., postgresql://user:pass@localhost/db")

    if st.button("Export to SQL"):
        engine = get_sql_engine(sql_uri)

        with (
            st.spinner("Exporting..."),
            engine.connect() as connection,
        ):
            export.export_sql(forest, connection)

        st.success("Exported successfully.")


def _render_graph_export(forest: TreeBucket) -> None:
    col1, col2, col3 = st.columns(3)
    graph_uri = col1.text_input("Graph URI", value="bolt://localhost:7687")
    username = col2.text_input("Username", value="neo4j")
    password = col3.text_input("Password", type="password")

    if st.button("Export to Graph"):
        driver = get_neo4j_driver(graph_uri, username=username, password=password)

        with (
            st.spinner("Exporting..."),
            driver.session() as session,
        ):
            export.export_cypher(forest, session)

        st.success("Exported successfully.")


def _render_jsonl_export(forest: TreeBucket) -> None:
    def get_jsonl() -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "architxt_export.jsonl"
            export_forest_to_jsonl(tmp_path, forest)
            return tmp_path.read_bytes()

    st.download_button(
        label="Export & Download JSONL",
        data=get_jsonl,
        file_name="architxt_export.jsonl",
        mime="application/jsonlines",
    )


EXPORTER = {
    "JSONL File": _render_jsonl_export,
    "SQL Database": _render_sql_export,
    "Graph Database": _render_graph_export,
}


@st.fragment
def exporter() -> None:
    st.header("Export Data")

    forest = get_forest()

    if len(forest) == 0:
        st.info("No data to export.")

    else:
        export_format = st.selectbox("Format", EXPORTER.keys())
        EXPORTER[export_format](forest)

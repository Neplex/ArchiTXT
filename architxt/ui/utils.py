import streamlit as st
from neo4j import Driver, GraphDatabase
from sqlalchemy import Engine, create_engine

from architxt.bucket.zodb import ZODBTreeBucket
from architxt.schema import Schema

METRIC_BEFORE_KEY = "prev_metrics"
METRIC_KEY = "metrics"
DEFAULT_METRIC: dict[str, int] = {
    "Total Trees": 0,
    "Entities": 0,
    "Groups": 0,
    "Relations": 0,
}


def get_metrics() -> tuple[dict[str, int], dict[str, int]]:
    return (
        st.session_state.get(METRIC_BEFORE_KEY, DEFAULT_METRIC).copy(),
        st.session_state.get(METRIC_KEY, DEFAULT_METRIC).copy(),
    )


def update_metrics() -> None:
    forest = get_forest()

    get_schema.clear()
    schema = get_schema()

    metrics = {
        "Total Trees": len(forest),
        "Entities": len(schema.entities),
        "Groups": len(schema.groups),
        "Relations": len(schema.relations),
    }

    st.session_state[METRIC_BEFORE_KEY] = st.session_state.get(METRIC_KEY, DEFAULT_METRIC)
    st.session_state[METRIC_KEY] = metrics
    st.session_state.pop("group_renames", None)
    st.session_state.pop("relation_renames", None)

    st.rerun()


def clear_data() -> None:
    forest = get_forest()

    with forest.transaction():
        forest.clear()

    update_metrics()


@st.cache_resource(scope="session", on_release=lambda bucket: bucket.close())
def get_forest() -> ZODBTreeBucket:
    return ZODBTreeBucket()


@st.cache_data(scope="session")
def get_schema() -> Schema:
    forest = get_forest()
    return Schema.from_forest(forest)


@st.cache_resource(scope="session", on_release=lambda engine: engine.dispose())
def get_sql_engine(sql_uri: str) -> Engine:
    return create_engine(sql_uri)


@st.cache_resource(scope="session", on_release=lambda driver: driver.close())
def get_neo4j_driver(graph_uri: str, *, username: str | None = None, password: str | None = None) -> Driver:
    auth = (username, password) if username and password else None
    return GraphDatabase.driver(graph_uri, auth=auth)

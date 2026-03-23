import streamlit as st
from neo4j import Driver, GraphDatabase
from sqlalchemy import Engine, create_engine

from architxt.bucket.zodb import ZODBTreeBucket


@st.cache_resource(scope="session", on_release=lambda bucket: bucket.close())
def get_forest() -> ZODBTreeBucket:
    return ZODBTreeBucket()


@st.cache_resource(scope="session")
def get_sql_engine(sql_uri: str) -> Engine:
    return create_engine(sql_uri)


@st.cache_resource(scope="session", on_release=lambda driver: driver.close())
def get_neo4j_driver(graph_uri: str, *, username: str | None = None, password: str | None = None) -> Driver:
    auth = (username, password) if username and password else None
    return GraphDatabase.driver(graph_uri, auth=auth)

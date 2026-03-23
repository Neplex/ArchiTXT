import pandas as pd
import streamlit as st
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node

from architxt.schema import Schema
from architxt.tree import Forest
from architxt.ui.utils import get_forest

MAX_LINE = 1_000


class Node(_Node):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f'Node({self.id})'


class Edge(_Edge):
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.source == other.source
            and self.to == other.to
            and getattr(self, "label", None) == getattr(other, "label", None)
        )

    def __hash__(self) -> int:
        return hash((self.source, self.to, getattr(self, "label", None)))

    def __repr__(self) -> str:
        return f'Edge({self.source}, {self.to}, label={getattr(self, "label", None)})'


@st.fragment
def render_schema(schema: Schema) -> None:
    """Render schema graph visualization."""
    nodes = set()
    edges = set()

    for entity in schema.entities:
        nodes.add(Node(id=entity, label=entity))

    for group in schema.groups:
        nodes.add(Node(id=group.name, label=group.name))

        for entity in group.entities:
            edges.add(Edge(source=group.name, target=entity))

    for relation in schema.relations:
        edges.add(Edge(source=relation.left, target=relation.right, label=relation.name))

    agraph(nodes=nodes, edges=edges, config=Config(directed=True))


@st.fragment
def render_dataframe(forest: Forest, schema: Schema) -> None:
    """Render instance DataFrames."""
    if not schema.groups:
        st.warning("No groups found in the forest.")
        return

    group_name = st.selectbox('Group', sorted(group.name for group in schema.groups))
    dataset = pd.DataFrame()

    with st.spinner("Loading dataset..."):
        for tree in forest:
            tree_dataset = tree.group_instances(group_name)
            dataset = pd.concat([dataset, tree_dataset], ignore_index=True).drop_duplicates()
            if len(dataset) > MAX_LINE:
                break

    if len(dataset):
        st.dataframe(dataset, width='stretch')

    else:
        st.warning("No instances found for the selected group.")


def visualizer() -> None:
    c1, c2 = st.columns([1, 2])

    forest = get_forest()
    forest.sync()

    schema = Schema.from_forest(forest)

    with c1:
        st.header("Schema")
        render_schema(schema)

    with c2:
        st.header("Instance")
        render_dataframe(forest, schema)

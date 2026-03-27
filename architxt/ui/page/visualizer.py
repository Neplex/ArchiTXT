import pandas as pd
import streamlit as st
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node

from architxt.schema import RelationOrientation, Schema
from architxt.tree import Forest
from architxt.ui.utils import get_forest, get_schema

MAX_LINE = 100


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
        nodes.add(Node(id=f'ENT::{entity}', label=entity, color="#4444FF"))

    for group in schema.groups:
        nodes.add(Node(id=f'GROUP::{group.name}', label=group.name, color="#FF4444"))

        for entity in group.entities:
            edges.add(Edge(source=f'GROUP::{group.name}', target=f'ENT::{entity}'))

    for relation in schema.relations:
        left = f'GROUP::{relation.left}'
        right = f'GROUP::{relation.right}'

        if relation.orientation == RelationOrientation.RIGHT:
            edge = Edge(source=right, target=left, label=relation.name)
        elif relation.orientation == RelationOrientation.LEFT:
            edge = Edge(source=left, target=right, label=relation.name)
        else:
            edge = Edge(source=left, target=right, label=relation.name, arrows={"to": True, "from": True})

        edges.add(edge)

    config = Config(directed=True, hierarchical=True)
    agraph(nodes=nodes, edges=edges, config=config)


@st.fragment
def render_dataframe(forest: Forest, groups: set[str]) -> None:
    """Render instance DataFrames."""
    group_name = st.selectbox('Group', sorted(groups))
    dataset = pd.DataFrame()

    with st.spinner("Loading dataset..."):
        for tree in forest:
            tree_dataset = tree.group_instances(group_name)
            dataset = pd.concat([dataset, tree_dataset], ignore_index=True).drop_duplicates()
            if len(dataset) > MAX_LINE:
                break

    if len(dataset):
        cols = sorted(dataset.columns, key=lambda c: (dataset[c].isna().sum(), c))
        row_nulls = dataset[cols].isna().sum(axis=1)

        dataset = (
            dataset.assign(_row_nulls=row_nulls)
            .sort_values(
                by=['_row_nulls', *cols],
                ascending=True,
                na_position='last',
                kind='mergesort',
            )
            .drop(columns='_row_nulls')
            .reindex(columns=cols)
            .reset_index(drop=True)
        )

        st.dataframe(dataset, width='stretch')

    else:
        st.warning("No instances found for the selected group.")


@st.fragment
def visualizer() -> None:
    c1, c2 = st.columns([1, 2])

    forest = get_forest()
    schema = get_schema()

    with c1:
        st.header("Schema")
        render_schema(schema)

    with c2:
        st.header("Instance")

        if groups := {group.name for group in schema.groups}:
            render_dataframe(forest, groups)
        else:
            st.warning("No groups found in the forest.")

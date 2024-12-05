from copy import deepcopy

import mlflow
import streamlit as st
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node
from streamlit_tags import st_tags

from architxt.algo import rewrite
from architxt.cli import load_or_cache_corpus
from architxt.db import Schema
from architxt.model import NodeType
from architxt.tree import Forest, Tree, has_type

mlflow.set_experiment('ArchiTXT')


class Node(_Node):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f'Node({self.id})'


class Edge(_Edge):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.source == other.source and self.to == other.to

    def __hash__(self) -> int:
        return hash((self.source, self.to))

    def __repr__(self) -> str:
        return f'Edge({self.source}, {self.to})'


@st.fragment()
def graph(forest: Forest) -> None:
    """Function to render schema graph visualization"""
    nodes = set()
    edges = set()

    schema = Schema.from_forest(forest, keep_invalid_nodes=False)
    for prod in schema.productions():
        if has_type(prod, {NodeType.GROUP, NodeType.REL}):
            lhs_symbol = prod.lhs().symbol().name
            nodes.add(Node(id=lhs_symbol, label=lhs_symbol))

            for nt in prod.rhs():
                symbol = nt.symbol().name if isinstance(nt.symbol(), NodeType) else nt.symbol()
                nodes.add(Node(id=symbol, label=symbol))

                label = 'REL' if has_type(prod, NodeType.REL) else ''
                edges.add(Edge(source=lhs_symbol, target=symbol, label=label))

    agraph(nodes=nodes, edges=edges, config=Config(directed=True))


@st.fragment()
def dataframe(forest: Forest) -> None:
    """Function to render instance DataFrames"""
    final_tree = Tree('ROOT', deepcopy(forest))
    table = st.selectbox('Group', sorted(final_tree.groups()))
    st.dataframe(final_tree.group_instances(table))


st.title("ArchiTxt")

with st.sidebar:
    corenlp_url = st.text_input('Corenlp URL', value='http://localhost:9000')
    language = st.selectbox('Language', ['French', 'English'])

input_tab, stats_tab, schema_tab, instance_tab = st.tabs(['📖 Corpus', '📊 Statistics', '📐 Schema', '🗄️ Instance'])

with input_tab, st.form(key='corpora', enter_to_submit=False):
    uploaded_file = st.file_uploader('Corpora', ['.tar.gz', '.tar.xz'], accept_multiple_files=False)

    entities_filter = st_tags(label='Excluded entities', value=['MOMENT', 'DUREE', 'DATE'])
    relations_filter = st_tags(label='Excluded relations', value=['TEMPORALITE', 'CAUSE-CONSEQUENCE'])

    st.divider()

    col1, col2, col3 = st.columns(3)
    tau = col1.number_input('Tau', min_value=0.05, max_value=1.0, step=0.05, value=0.5)
    epoch = col2.number_input('Epoch', min_value=1, step=1, value=100)
    min_support = col3.number_input('Minimum Support', min_value=1, step=1, value=10)
    submitted = st.form_submit_button("Start")

if submitted and uploaded_file:
    try:
        forest = load_or_cache_corpus(
            uploaded_file,
            entities_filter=set(entities_filter),
            relations_filter=set(relations_filter),
            entities_mapping={'FREQ': 'FREQUENCE'},
            corenlp_url=corenlp_url,
            language=language,
        )

        with st.spinner('Computing...'), mlflow.start_run(run_name='UI run', log_system_metrics=True) as mlflow_run:
            database = rewrite(
                forest,
                tau=tau,
                epoch=epoch,
                min_support=min_support,
            )

            # Display statistics tab
            with stats_tab:
                run_id = mlflow_run.info.run_id
                client = mlflow.tracking.MlflowClient()

                st.line_chart(
                    {
                        metric: [x.value for x in client.get_metric_history(run_id, metric)]
                        for metric in [
                            'num_productions',
                            'num_unlabeled_nodes',
                            'num_groups',
                            'num_relations',
                            'num_collections',
                        ]
                    }
                )

                st.bar_chart([x.value for x in client.get_metric_history(run_id, 'edit_op')])

            # Display schema graph
            with schema_tab:
                graph(database)

            # Display instance data
            with instance_tab:
                dataframe(database)

    except Exception as e:
        st.error(f"An error occurred: {e!s}")

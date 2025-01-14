from copy import deepcopy

import mlflow
import streamlit as st
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node
from streamlit_tags import st_tags

from architxt.cli import load_or_cache_corpus
from architxt.model import NodeType
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite
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
def graph(schema: Schema) -> None:
    """Function to render schema graph visualization"""
    nodes = set()
    edges = set()

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

input_tab, stats_tab, schema_tab, instance_tab = st.tabs(['📖 Corpus', '📊 Metrics', '📐 Schema', '🗄️ Instance'])

with input_tab:
    uploaded_file = st.file_uploader('Corpora', ['.tar.gz', '.tar.xz'], accept_multiple_files=True)

    file_language = []
    for file in uploaded_file:
        language_columns = st.columns(2)
        language_columns[0].text_input('Corpus', file.name, disabled=True)
        language = language_columns[1].selectbox('Language', ['French', 'English'], key=file.name)
        file_language.append((file, language))

    st.divider()

    with st.form(key='corpora', enter_to_submit=False):
        entities_filter = st_tags(label='Excluded entities', value=['MOMENT', 'DUREE', 'DATE'])
        relations_filter = st_tags(label='Excluded relations', value=['TEMPORALITE', 'CAUSE-CONSEQUENCE'])

        st.divider()

        col1, col2, col3 = st.columns(3)
        tau = col1.number_input('Tau', min_value=0.05, max_value=1.0, step=0.05, value=0.5)
        epoch = col2.number_input('Epoch', min_value=1, step=1, value=100)
        min_support = col3.number_input('Minimum Support', min_value=1, step=1, value=10)
        submitted = st.form_submit_button("Start")

if submitted and file_language:
    try:
        forest = []
        for file, language in file_language:
            forest += load_or_cache_corpus(
                file,
                entities_filter=set(entities_filter),
                relations_filter=set(relations_filter),
                entities_mapping={'FREQ': 'FREQUENCE'},
                corenlp_url=corenlp_url,
                language=language,
            )

        if mlflow.active_run():
            mlflow.end_run()

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
                        'coverage',
                        'similarity',
                        'edit_distance',
                        'cluster_ami',
                        'cluster_completeness',
                        'overlap',
                    ]
                }
            )

            st.line_chart(
                {
                    metric: [x.value for x in client.get_metric_history(run_id, metric)]
                    for metric in [
                        'num_productions',
                        'unlabeled_nodes',
                        'group_instance_total',
                        'relation_instance_total',
                        'collection_instance_total',
                    ]
                }
            )

            st.bar_chart([x.value for x in client.get_metric_history(run_id, 'edit_op')])

        schema = Schema.from_forest(forest, keep_unlabelled=False)

        # Display schema graph
        with schema_tab:
            graph(schema)

        # Display instance data
        with instance_tab:
            clean_database = schema.extract_valid_trees(database)
            dataframe(clean_database)

    except Exception as e:
        st.error(f"An error occurred: {e!s}")

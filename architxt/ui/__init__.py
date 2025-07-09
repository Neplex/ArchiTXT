import random
from copy import deepcopy

import anyio
import mlflow
import pandas as pd
import streamlit as st
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node
from streamlit_tags import st_tags

from architxt.cli.loader import ENTITIES_FILTER, ENTITIES_MAPPING, RELATIONS_FILTER
from architxt.nlp import raw_load_corpus
from architxt.nlp.parser.corenlp import CoreNLPParser
from architxt.schema import Schema
from architxt.simplification.tree_rewriting import rewrite
from architxt.tree import Forest, Tree

RESOLVER_NAMES = {
    None: 'No resolution',
    'umls': 'Unified Medical Language System (UMLS)',
    'mesh': 'Medical Subject Headings (MeSH)',
    'rxnorm': 'RxNorm',
    'go': 'Gene Ontology (GO)',
    'hpo': 'Human Phenotype Ontology (HPO)',
}


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


@st.fragment()
def dataframe(forest: Forest) -> None:
    """Render instance DataFrames."""
    final_tree = Tree('ROOT', deepcopy(forest))
    group_name = st.selectbox('Group', sorted(final_tree.groups()))
    st.dataframe(final_tree.group_instances(group_name), use_container_width=True)


st.title("ArchiTXT")

with st.sidebar:
    corenlp_url = st.text_input('Corenlp URL', value='http://localhost:9000')
    resolver_name = st.selectbox(
        'Entity Resolver',
        options=RESOLVER_NAMES.keys(),
        format_func=RESOLVER_NAMES.get,
    )

input_tab, stats_tab, schema_tab, instance_tab = st.tabs(['📖 Corpus', '📊 Metrics', '📐 Schema', '🗄️ Instance'])

with input_tab:
    uploaded_files = st.file_uploader('Corpora', ['.tar.gz', '.tar.xz'], accept_multiple_files=True)

    if uploaded_files:
        file_language_table = st.data_editor(
            pd.DataFrame([{'Corpora': file.name, 'Language': 'English'} for file in uploaded_files]),
            column_config={
                'Corpora': st.column_config.TextColumn(disabled=True),
                'Language': st.column_config.SelectboxColumn(options=['English', 'French'], required=True),
            },
            hide_index=True,
            use_container_width=True,
        )
        file_language = {row['Corpora']: row['Language'] for _, row in file_language_table.iterrows()}

    st.divider()

    with st.form(key='corpora', enter_to_submit=False):
        entities_filter = st_tags(label='Excluded entities', value=list(ENTITIES_FILTER))
        relations_filter = st_tags(label='Excluded relations', value=list(RELATIONS_FILTER))
        st.text('Entity mapping')
        entity_mapping = st.data_editor(ENTITIES_MAPPING, use_container_width=True, hide_index=True, num_rows="dynamic")

        st.divider()

        col1, col2, col3 = st.columns(3)
        tau = col1.number_input('Tau', min_value=0.05, max_value=1.0, step=0.05, value=0.5)
        epoch = col2.number_input('Epoch', min_value=1, step=1, value=100)
        min_support = col3.number_input('Minimum Support', min_value=1, step=1, value=10)
        sample = col1.number_input('Sample size', min_value=0, step=1, value=0, help='0 means no sampling')
        shuffle = col2.selectbox('Shuffle', options=[True, False])
        submitted = st.form_submit_button("Start")


async def load_forest() -> list[Tree]:
    if not uploaded_files:
        return []

    languages = [file_language[file.name] for file in uploaded_files]

    return await raw_load_corpus(
        uploaded_files,
        languages,
        entities_filter=set(entities_filter),
        relations_filter=set(relations_filter),
        entities_mapping=entity_mapping,
        parser=CoreNLPParser(corenlp_url=corenlp_url),
        resolver_name=resolver_name,
    )


if submitted and file_language:
    try:
        if mlflow.active_run():
            mlflow.end_run()

        with st.spinner('Computing...'), mlflow.start_run(description='UI run', log_system_metrics=True) as mlflow_run:
            forest = anyio.run(load_forest)

            if sample:
                forest = random.sample(forest, sample)

            if shuffle:
                random.shuffle(forest)

            rewrite(
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
                pd.DataFrame(
                    [
                        metric.to_dictionary()
                        for metric_name in [
                            'coverage',
                            'cluster_ami',
                            'cluster_completeness',
                            'overlap',
                            'balance',
                        ]
                        for metric in client.get_metric_history(run_id, metric_name)
                    ]
                ),
                x='step',
                y='value',
                color='key',
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
            dataframe(forest)

    except Exception as e:
        st.error(f"An error occurred: {e!s}")

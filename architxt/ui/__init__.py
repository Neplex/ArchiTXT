import hashlib
from copy import deepcopy
from pathlib import Path
from tarfile import TarFile
from tempfile import TemporaryDirectory

import mlflow
import streamlit as st
from ray import cloudpickle
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_agraph import Config, agraph
from streamlit_agraph import Edge as _Edge
from streamlit_agraph import Node as _Node
from streamlit_tags import st_tags

from architxt.algo import rewrite
from architxt.model import NodeType
from architxt.nlp import get_enriched_forest, get_sentence_from_disk
from architxt.tree import Forest, Tree, has_type

mlflow.set_experiment('ArchiTXT')


class Node(_Node):
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f'Node({self.id})'


class Edge(_Edge):
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.source == other.source and self.to == other.to

    def __hash__(self):
        return hash((self.source, self.to))

    def __repr__(self):
        return f'Edge({self.source}, {self.to})'


def get_forest(archive_file: UploadedFile, *, entities_filter: set[str], relations_filter: set[str]) -> Forest:
    key = hashlib.md5(
        (archive_file.name + 'E'.join(sorted(entities_filter)) + 'R'.join(sorted(relations_filter))).encode()
    ).hexdigest()
    corpus_cache_path = Path() / f'{key}.pkl'

    if corpus_cache_path.exists():
        print(f'Loading corpus from cache: {corpus_cache_path.absolute()}')
        with open(corpus_cache_path, 'rb') as cache_file:
            forest = cloudpickle.load(cache_file)

    else:
        print('Loading corpus...')
        with TemporaryDirectory() as tmp_dir, TarFile.open(fileobj=archive_file) as corpus:
            corpus.extractall(tmp_dir)
            sentences = get_sentence_from_disk(
                Path(tmp_dir),
                entities_filter=entities_filter,
                relations_filter=relations_filter,
                entities_mapping={'FREQ': 'FREQUENCE'},
            )

            forest = tuple(get_enriched_forest(sentences, corenlp_url=corenlp_url, language=language))

        print(f'Saving cache file to: {corpus_cache_path.absolute()}')
        with open(corpus_cache_path, 'wb') as cache_file:
            cloudpickle.dump(forest, cache_file)

    print(f'Dataset loaded! {len(forest)} sentences found')
    return forest


@st.fragment()
def graph(forest: Forest):
    """Function to render schema graph visualization"""
    nodes = set()
    edges = set()

    for tree in forest:
        for prod in tree.productions():
            if prod.is_nonlexical() and has_type(prod.lhs().symbol(), {NodeType.GROUP, NodeType.REL}):
                lhs_symbol = prod.lhs().symbol().name
                nodes.add(Node(id=lhs_symbol, label=lhs_symbol))

                for nt in prod.rhs():
                    symbol = nt.symbol().name
                    nodes.add(Node(id=symbol, label=symbol))

                    label = 'REL' if prod.lhs().symbol().type == NodeType.REL else ''
                    edges.add(Edge(source=lhs_symbol, target=symbol, label=label))

    agraph(nodes=nodes, edges=edges, config=Config(directed=True))


@st.fragment()
def dataframe(forest: Forest):
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
    uploaded_file = st.file_uploader('Corpora', ['.tar.gz', '.tar.xz'], False)

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
        forest = get_forest(
            uploaded_file,
            entities_filter=set(entities_filter),
            relations_filter=set(relations_filter),
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

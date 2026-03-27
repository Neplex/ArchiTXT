from collections.abc import Generator
from functools import partial

import anyio
import pandas as pd
import streamlit as st
from spacy.util import get_installed_models
from streamlit_tags import st_tags

from architxt.bucket import TreeBucket
from architxt.cli.loader import ENTITIES_FILTER, ENTITIES_MAPPING, RELATIONS_FILTER
from architxt.database import loader
from architxt.database.loader import read_document
from architxt.forest import import_forest_from_jsonl
from architxt.nlp import raw_load_corpus
from architxt.nlp.entity_resolver import ScispacyResolver
from architxt.nlp.parser.benepar import BeneparParser
from architxt.nlp.parser.corenlp import CoreNLPParser
from architxt.tree import Tree
from architxt.ui.utils import get_forest, get_neo4j_driver, get_sql_engine, update_metrics

RESOLVER_NAMES = {
    None: 'No resolution',
    'umls': 'Unified Medical Language System (UMLS)',
    'mesh': 'Medical Subject Headings (MeSH)',
    'rxnorm': 'RxNorm',
    'go': 'Gene Ontology (GO)',
    'hpo': 'Human Phenotype Ontology (HPO)',
}


@st.cache_resource
def get_corenlp_parser(corenlp_url: str) -> CoreNLPParser:
    return CoreNLPParser(corenlp_url=corenlp_url)


@st.cache_resource
def get_benepar_parser(language: str, spacy_model: str) -> BeneparParser:
    return BeneparParser(spacy_models={language: spacy_model})


def _render_text_corpus_import(forest: TreeBucket) -> None:
    uploaded_files = st.file_uploader('Corpora', ['.tar.gz', '.tar.xz', '.txt'], accept_multiple_files=True)

    col1, col2, col3 = st.columns(3)
    language = col1.selectbox("Language", ['English', 'French', 'German'])
    parser_type = col2.selectbox("Parser", ["Benepar", "CoreNLP"])

    if parser_type == "Benepar":
        spacy_model = col3.selectbox("Spacy Model", get_installed_models())
        parser = get_benepar_parser(language, spacy_model)
    else:
        corenlp_url = col3.text_input('CoreNLP URL', value='http://localhost:9000')
        parser = get_corenlp_parser(corenlp_url)

    col1, col2 = st.columns(2)
    resolver_name = col1.selectbox('Entity Resolver', options=RESOLVER_NAMES.keys(), format_func=RESOLVER_NAMES.get)
    sample = col2.number_input('Sample', min_value=1, value=None)

    with st.expander("Advanced NLP Settings"):
        entities_filter = st_tags(label='Excluded entities', value=list(ENTITIES_FILTER))
        relations_filter = st_tags(label='Excluded relations', value=list(RELATIONS_FILTER))
        st.text('Entity mapping')

        # Convert dict to DataFrame for editing
        mapping_df = pd.DataFrame(ENTITIES_MAPPING.items(), columns=['From', 'To'])
        edited_mapping_df = st.data_editor(mapping_df, width='stretch', num_rows="dynamic").dropna()
        # Convert back to dict
        entity_mapping = dict(zip(edited_mapping_df['From'], edited_mapping_df['To']))

    if st.button("Load Corpus", disabled=not uploaded_files):
        resolver = None
        if resolver_name:
            try:
                resolver = ScispacyResolver(kb_name=resolver_name)
            except Exception as e:
                st.warning(f"Could not initialize resolver: {e}")

        trees = raw_load_corpus(
            uploaded_files,
            [language] * len(uploaded_files),
            entities_filter=set(entities_filter),
            relations_filter=set(relations_filter),
            entities_mapping=entity_mapping,
            parser=parser,
            resolver=resolver,
            sample=sample,
            cache=st.session_state.get("cache", True),
        )

        with st.spinner("Parsing and Loading Corpus..."):
            forest_update = partial(forest.async_update, commit=True)
            anyio.run(forest_update, trees)

        update_metrics()


def _render_sql_import(forest: TreeBucket) -> None:
    sql_uri = st.text_input("SQL URI", value="sqlite:///example.db", help="e.g., postgresql://user:pass@localhost/db")

    if st.button("Load SQL"):
        engine = get_sql_engine(sql_uri)

        with (
            st.spinner("Loading from SQL..."),
            engine.connect() as connection,
        ):
            trees = loader.read_sql(connection)
            forest.update(trees, commit=True)

        update_metrics()


def _render_graph_import(forest: TreeBucket) -> None:
    col1, col2, col3 = st.columns(3)
    graph_uri = col1.text_input("Graph URI", value="bolt://localhost:7687")
    username = col2.text_input("Username", value="neo4j")
    password = col3.text_input("Password", type="password")

    if st.button("Load Graph"):
        driver = get_neo4j_driver(graph_uri, username=username, password=password)

        with (
            st.spinner("Loading from Graph..."),
            driver.session() as session,
        ):
            trees = loader.read_cypher(session)
            forest.update(trees, commit=True)

        update_metrics()


def _render_document_import(forest: TreeBucket) -> None:
    uploaded_documents = st.file_uploader(
        "Document File",
        [".json", ".toml", ".yml", ".yaml", ".xml", ".csv", ".xls", ".xlsx"],
        accept_multiple_files=True,
    )

    if st.button("Load document", disabled=not uploaded_documents):
        with st.spinner("Loading from document..."):

            def load_trees() -> Generator[Tree, None, None]:
                for document in uploaded_documents:
                    yield from read_document(document, root_name=document.name)

            forest.update(load_trees(), commit=True)

        update_metrics()


def _render_jsonl_import(forest: TreeBucket) -> None:
    uploaded_jsonls = st.file_uploader("JSONL File", ["jsonl"], accept_multiple_files=True)

    if st.button("Load JSONL", disabled=not uploaded_jsonls):
        with st.spinner("Loading from JSONL..."):

            def load_trees() -> Generator[Tree, None, None]:
                for jsonl in uploaded_jsonls:
                    yield from import_forest_from_jsonl(jsonl)

            forest.update(load_trees(), commit=True)

        update_metrics()


IMPORTER = {
    "JSONL File": _render_jsonl_import,
    "Document File": _render_document_import,
    "SQL Database": _render_sql_import,
    "Graph Database": _render_graph_import,
    "Text Corpus": _render_text_corpus_import,
}


@st.fragment
def importer() -> None:
    st.header("Import Data")

    forest = get_forest()

    source_type = st.selectbox("Source Type", IMPORTER.keys())
    IMPORTER[source_type](forest)

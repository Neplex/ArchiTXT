import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import mlflow
import typer
from nltk import Production
from ray import cloudpickle

from architxt.algo import rewrite
from architxt.generator import gen_instance
from architxt.model import NodeLabel, NodeType
from architxt.nlp import get_enriched_forest, get_sentence_from_disk
from architxt.tree import Tree


def cli_run(
    corpus_path: Path,
    *,
    tau: float = 0.5,
    epoch: int = 100,
    min_support: int = 10,
    corenlp_url: str = 'http://localhost:9000',
    gen_instances: int = 0,
    language: str = 'French',
    debug: bool = False,
) -> None:
    mlflow.log_params(
        {
            'has_corpus': True,
            'has_instance': bool(gen_instances),
        }
    )

    entities_filter = {'MOMENT', 'DUREE', 'DATE'}
    relations_filter = {'TEMPORALITE', 'CAUSE-CONSEQUENCE'}

    # Load the corpus from the cache if available, else load it from the disk
    key = hashlib.md5(
        (corpus_path.name + 'E'.join(sorted(entities_filter)) + 'R'.join(sorted(relations_filter))).encode()
    ).hexdigest()
    corpus_cache_path = Path() / f'{key}.pkl'

    if corpus_cache_path.exists():
        print(f'Load corpus from cache: {corpus_cache_path.absolute()}')
        with open(corpus_cache_path, 'rb') as cache_file:
            forest = cloudpickle.load(cache_file)

    else:
        print(f'Load corpus from disk: {corpus_path.absolute()}')
        sentences = get_sentence_from_disk(
            corpus_path,
            entities_filter={'MOMENT', 'DUREE', 'DATE'},
            relations_filter={'TEMPORALITE', 'CAUSE-CONSEQUENCE'},
            entities_mapping={'FREQ': 'FREQUENCE'},
        )
        forest = tuple(get_enriched_forest(sentences, corenlp_url=corenlp_url, language=language))

        print(f'Save cache file to: {corpus_cache_path.absolute()}')
        with open(corpus_cache_path, 'wb') as cache_file:
            cloudpickle.dump(forest, cache_file)

    print(f'Dataset loaded! {len(forest)} sentences found')

    # Generate a synthetic database instance
    if gen_instances:
        print('Generate instance...')
        gen_trees = gen_instance(
            groups={
                'SOSY': ('SOSY', 'ANATOMIE', 'SUBSTANCE'),
                'TREATMENT': ('SUBSTANCE', 'DOSE', 'MODE', 'FREQUENCE'),
                'EXAM': ('EXAMEN', 'ANATOMIE'),
            },
            rels={
                'PRESCRIPTION': ('SOSY', 'TREATMENT'),
                'EXAM': ('EXAM', 'SOSY'),
            },
            size=gen_instances,
        )
        forest.extend(gen_trees)

    # Rewrite tree
    forest = rewrite(
        forest,
        tau=tau,
        epoch=epoch,
        min_support=min_support,
        debug=debug,
    )
    print('Done!')

    final_tree = Tree('ROOT', forest)

    productions = []
    schema = {}
    for prod in final_tree.productions():
        if prod.is_nonlexical():
            if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type == NodeType.COLL:
                productions.append(Production(prod.lhs(), [prod.rhs()[0]]))
                schema[prod.lhs()] = [str(prod.rhs()[0]) + '*']
            else:
                productions.append(Production(prod.lhs(), sorted(prod.rhs())))

                if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type in (
                    NodeType.GROUP,
                    NodeType.REL,
                ):
                    old = set(schema[prod.lhs()]) if prod.lhs() in schema else set()
                    new = {str(x) for x in prod.rhs()}

                    schema[prod.lhs()] = sorted(old | new)

    schema_str = "\n".join(
        f"{key} -> {', '.join(value)}"
        for key, value in sorted(schema.items(), key=lambda x: (x[0].symbol().type, x[0].symbol().name))
    )

    print(schema_str)
    mlflow.log_text(schema_str, 'schema.txt')

    print('\n' + '=' * 10 + '\n')

    for production, count in Counter(productions).most_common():
        if production.is_nonlexical():
            print(f'[{count}] {production}')


def cli_ui() -> None:
    from .. import ui

    subprocess.run(['streamlit', 'run', ui.__file__])


def main() -> None:
    """Main function"""
    mlflow.set_experiment('ArchiTXT')

    app = typer.Typer()
    app.command('run')(cli_run)
    app.command('ui')(cli_ui)
    app()

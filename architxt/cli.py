from collections import Counter
from pathlib import Path

import mlflow
import typer
from nltk import Production

from architxt.algo import rewrite
from architxt.model import NodeType, NodeLabel
from architxt.nlp import get_sentence_from_disk, get_annotated_rooted_forest


def cli(
        corpus_path: Path, *,
        tau: float = 0.5,
        epoch: int = 100,
        min_support: int = 10
):
    mlflow.log_params({
        'has_instance': False,
        'has_corpus': True,
    })

    # print('Generate instance...')
    # gen_tree = gen_instance(
    #     groups={
    #         'SOSY': ('SOSY', 'ANATOMIE', 'SUBSTANCE'),
    #         'TREATMENT': ('SUBSTANCE', 'DOSE', 'MODE', 'FREQUENCE'),
    #         'EXAM': ('EXAMEN', 'ANATOMIE'),
    #     },
    #     rels={
    #         'PRESCRIPTION': ('SOSY', 'TREATMENT'),
    #         'EXAM': ('EXAM', 'SOSY'),
    #     },
    #     size=50
    # )

    print(f'Load corpus: {corpus_path.absolute()}')
    sentences = list(get_sentence_from_disk(corpus_path))[:150]

    corpus_tree = get_annotated_rooted_forest(sentences, url='http://localhost:9001')
    print('Dataset loaded!')

    tree = corpus_tree  # gen_tree.merge(corpus_tree)
    mlflow.log_param('nb_sentences', len(tree))
    with open('debug.txt', 'w', encoding='utf8') as log_file:
        rewrite(tree, tau=tau, epoch=epoch, min_support=min_support, stream=log_file)
    print('Done!')

    productions = []
    schema = {}
    for prod in tree.productions():
        if prod.is_nonlexical():
            if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type == NodeType.COLL:
                productions.append(Production(prod.lhs(), [prod.rhs()[0]]))
                schema[prod.lhs()] = [str(prod.rhs()[0]) + '*']
            else:
                productions.append(Production(prod.lhs(), list(sorted(prod.rhs()))))

                if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type in (NodeType.GROUP, NodeType.REL):
                    old = set(schema[prod.lhs()]) if prod.lhs() in schema else set()
                    new = set(str(x) for x in prod.rhs())

                    schema[prod.lhs()] = list(sorted(old | new))

    schema_str = ""
    for key, value in sorted(schema.items(), key=lambda x: (x[0].symbol().type, x[0].symbol().name)):
        schema_str += f"{key} -> {', '.join(value)}\n"

    print(schema_str)
    mlflow.log_text(schema_str, 'schema.txt')
    mlflow.log_artifact('debug.txt')
    mlflow.log_artifact('trace.txt')

    print('\n' + '=' * 10 + '\n')

    for production, count in Counter(productions).most_common():
        production: Production
        if production.is_nonlexical():
            print(f'[{count}] {production}')

    tree.draw()


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()

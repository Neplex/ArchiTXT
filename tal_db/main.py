import sys
from collections import Counter
from pathlib import Path

from nltk import Production

from .algo import rewrite
from .model import NodeType, NodeLabel
from .nlp import get_sentence_from_disk, get_annotated_rooted_forest
from .instance_generator import gen_instance
from .similarity import levenshtein
import mlflow

if __name__ == '__main__':
    path = Path(sys.argv[1])
    tau = float(sys.argv[2])
    epoch = int(sys.argv[3])
    min_support = int(sys.argv[4])

    mlflow.log_params({
        'has_instance': False,
        'has_corpus': True,
    })

    print('Generate instance...')
    gen_tree = gen_instance(
        groups={
            'SOSY': ('SOSY', 'ANATOMIE', 'SUBSTANCE'),
            'TREATMENT': ('SUBSTANCE', 'DOSE', 'MODE', 'FREQUENCE'),
            'EXAM': ('EXAMEN', 'ANATOMIE'),
        },
        rels={
            'PRESCRIPTION': ('SOSY', 'TREATMENT'),
            'EXAM': ('EXAM', 'SOSY'),
        },
        size=25
    )

    print(f'Load corpus: {path.absolute()}')
    sentences = list(get_sentence_from_disk(path))[:66]
    corpus_tree = get_annotated_rooted_forest(sentences, url='http://localhost:9001')
    print('Dataset loaded!')

    tree = corpus_tree #gen_tree.merge(corpus_tree)
    with open('debug.log', 'w', encoding='utf8') as log_file:
        rewrite(tree, tau=.7, epoch=25, min_support=10, metric=levenshtein, stream=log_file)
    print('Done!')

    productions = []
    schema = {}
    for prod in tree.productions():
        if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type == NodeType.COLL:
            productions.append(Production(prod.lhs(), [prod.rhs()[0]]))
            schema[prod.lhs()] = [str(prod.rhs()[0]) + '*']
        else:
            productions.append(Production(prod.lhs(), list(sorted(prod.rhs()))))

            if isinstance(prod.lhs().symbol(), NodeLabel) and prod.lhs().symbol().type in (NodeType.GROUP, NodeType.REL):
                old = set(schema[prod.lhs()]) if prod.lhs() in schema else set()
                new = set(str(x) for x in prod.rhs())

                schema[prod.lhs()] = list(sorted(old | new))

    for key, value in sorted(schema.items(), key=lambda x: (x[0].symbol().type, x[0].symbol().name)):
        print(f"{key} -> {', '.join(value)}")

    print('\n' + '=' * 10 + '\n')

    for production, count in Counter(productions).most_common():
        production: Production
        if production.is_nonlexical():
            print(f'[{count}] {production}')

# ArchiTXT: Text-to-Database Structuring Tool

**ArchiTXT** is a robust tool designed to convert unstructured textual data into structured formats that are ready for
database storage. It automates the generation of database schemas and creates corresponding data instances, simplifying
the integration of text-based information into database systems.

Working with unstructured text can be challenging when you need to store and query it in a structured database.
**ArchiTXT** bridges this gap by transforming raw text into organized, query-friendly structures. By automating both
schema generation and data instance creation, it streamlines the entire process of managing textual information in
databases.

## Installation

To install **ArchiTXT**, make sure you have Python 3.10+ and pip installed. Then, run:

```sh
pip install architxt
```

## Usage

**ArchiTXT** is built to work seamlessly with BRAT-annotated corpora that include pre-labeled named entities. It also
requires access to a CoreNLP server, which you can easily set up using the Docker configuration available in the source
repository.

```sh
$ architxt --help

 Usage: architxt [OPTIONS] COMMAND [ARGS]...

 ArchiTXT is a tool for structuring textual data into a valid database model.
 It is guided by a meta-grammar and uses an iterative process of tree rewriting.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                        │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation. │
│ --help                        Show this message and exit.                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ run   Extract a database schema form a corpus.                                                                 │
│ ui    Launch the web-based UI.                                                                                 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```sh
$ architxt run --help

 Usage: architxt run [OPTIONS] CORPUS_PATH

 Extract a database schema form a corpus.

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    corpus_path      PATH  Path to the input corpus. [default: None] [required]                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --tau                            FLOAT    The similarity threshold. [default: 0.7]                             │
│ --epoch                          INTEGER  Number of iteration for tree rewriting. [default: 100]               │
│ --min-support                    INTEGER  Minimum support for tree patterns. [default: 20]                     │
│ --corenlp-url                    TEXT     URL of the CoreNLP server. [default: http://localhost:9000]          │
│ --gen-instances                  INTEGER  Number of synthetic instances to generate. [default: 0]              │
│ --language                       TEXT     Language of the input corpus. [default: French]                      │
│ --debug            --no-debug             Enable debug mode for more verbose output. [default: no-debug]       │
│ --help                                    Show this message and exit.                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

To deploy the CoreNLP server using the source repository, you can use Docker Compose with the following command:

```sh
docker compose up -d
```

## Development

### Meta-Grammar

To regenerate the meta-grammar parser/lexer, use the following :

```sh
$ poetry run antlr4 -Dlanguage=Python3 metagrammar.g4 -o architxt/grammar
```

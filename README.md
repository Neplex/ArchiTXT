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

 Usage: architxt [OPTIONS] CORPUS_PATH

╭─ Arguments ──────────────────────────────────────────────────────────╮
│ *    corpus_path      PATH  [default: None] [required]               │
╰──────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────╮
│ --tau                  FLOAT    [default: 0.5]                       │
│ --epoch                INTEGER  [default: 2]                         │
│ --min-support          INTEGER  [default: 10]                        │
│ --corenlp-url          TEXT     [default: http://localhost:9000]     │
│ --gen-instances        INTEGER  [default: 0]                         │
│ --language             TEXT     [default: French]                    │
│ --help                          Show this message and exit.          │
╰──────────────────────────────────────────────────────────────────────╯
```

To deploy the CoreNLP server using the source repository, you can use Docker Compose with the following command:

```sh
docker compose up -d
```

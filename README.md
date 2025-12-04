# ArchiTXT: Text-to-Database Structuring Tool

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![PyPI - Status](https://img.shields.io/pypi/status/architxt)
[![PyPI - Version](https://img.shields.io/pypi/v/architxt)](https://pypi.org/project/architxt/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/architxt)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/neplex/architxt/python-build.yml)](https://github.com/Neplex/ArchiTXT/actions)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/Neplex/ArchiTXT/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/Neplex/ArchiTXT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15688157.svg)](https://doi.org/10.5281/zenodo.15688157)

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

For the development version, you can install it directly through GIT using

```sh
pip install git+https://github.com/Neplex/ArchiTXT.git
```

## Usage

**ArchiTXT** is built to work seamlessly with BRAT-annotated corpora that includes pre-labeled named entities.
It also requires access to a CoreNLP server, which you can set up using the Docker configuration available in
the source repository.

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
docker compose up -d corenlp
```

## Sponsors

This work has received support under the JUNON Program, with financial support from Région Centre-Val de Loire (France).

<a href="https://www.junon-cvl.fr">
  <img src="https://www.junon-cvl.fr/sites/websites/www.junon-cvl.fr/files/logos/2025-07/logo-junon-new.svg" width="200" alt="JUNON Program logo">
</a>

<a href="https://www.univ-orleans.fr">
  <img src="https://ent.univ-orleans.fr/pages-locales-uo/images/logo_univ.svg" width="200" alt="UO logo">
</a>

<a href="https://www.univ-orleans.fr/lifo/">
  <img src="https://www.univ-orleans.fr/lifo/themes/custom/bs5_lifo_theme/logo.svg" width="200" alt="LIFO logo">
</a>

# ArchiTXT: Text-to-Database Structuring Tool

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![PyPI - Status](https://img.shields.io/pypi/status/architxt)
[![PyPI - Version](https://img.shields.io/pypi/v/architxt)](https://pypi.org/project/architxt/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/architxt)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/neplex/architxt/python-build.yml)](https://github.com/Neplex/ArchiTXT/actions)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/Neplex/ArchiTXT/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/Neplex/ArchiTXT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15688157.svg)](https://doi.org/10.5281/zenodo.15688157)

**ArchiTXT** is a Python library and CLI tool that automatically converts unstructured text corpora into structured,
database-ready data. It infers database schemas directly from text and generates corresponding structured instances
using a meta-grammar and iterative tree-rewriting process.

**ArchiTXT** is designed for researchers, data engineers, and NLP practitioners who need a transparent and auditable
process to transform raw textual data into storable, queryable and machine-learning-ready datasets.

## Why ArchiTXT?

Working with unstructured text becomes complex when you need:
- Structured storage
- Queryable entities and relations
- Reproducible data modeling

**ArchiTXT** bridges this gap by:
- Discovering latent structural patterns in annotated corpora
- Automatically generating database schemas
- Producing structured instances aligned with the inferred schema
- Ensuring transparency through rule-based rewriting

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
It can parse the texts using either CoreNLP or SpaCy, depending on your preference and setup.
See the [documentation](https://neplex.github.io/ArchiTXT/importers/text.html) for more information.

For CoreNLP, it requires access to a CoreNLP server, which you can set up using the Docker Compose configuration
available in the source repository. To deploy it, you can use the following command:

```sh
docker compose up -d corenlp
```

After parsing the annotated texts into **ArchiTXT**'s internal representation, you can infer a database schema and instance based on
the annotated entities and generate structured instances accordingly.
See the [documentation](https://neplex.github.io/ArchiTXT/transformers/simplify.html) for more information.

The result can be exported as a relational or property graph database.
See the [documentation](https://neplex.github.io/ArchiTXT/exporters.html) for more information.

**ArchiTXT** is available as a Python library but also provides a command-line interface (CLI) for users who prefer
working in the terminal. You can run the CLI using:

```sh
architxt --help
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

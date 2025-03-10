# Contributing to ArchiTXT

Thank you for considering contributing to **ArchiTXT**! Your involvement helps enhance the project and benefits the entire community. We welcome contributions in various forms, including bug reports, feature suggestions, code improvements, and documentation enhancements.

## Ways to Contribute

You can contribute in several ways:

- **Reporting Bugs:** Encountered an issue? Please report it so we can address it promptly.
- **Suggesting Features:** Have an idea for a new feature or improvement? We'd love to hear your thoughts.
- **Submitting Code:** If you're interested in adding features or fixing bugs, consider submitting a pull request.
- **Improving Documentation:** Help us make our documentation clearer and more comprehensive.

## Development Setup

To contribute code or documentation, set up the development environment as follows:

### Prerequisites

- **Python:** Ensure Python 3.10 or higher is installed.
- **Poetry:** We use [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

### Setting Up the Development Environment

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/neplex/architxt.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd architxt
    ```

3. **Install Dependencies:**

    ```bash
    poetry install
    ```

## Code Quality and Testing

We maintain high code quality and comprehensive testing through the following tools:

- **[Ruff](https://github.com/astral-sh/ruff)** for code quality and formatting.
- **[Pytest](https://docs.pytest.org/en/stable/)** to run unit tests.
- **[Hypothesis](https://hypothesis.readthedocs.io/en/latest/):** Integrated with Pytest, for property-based testing.

Both code quality checks and tests are enforced through our Continuous Integration (CI) pipeline.

### Pre-Commit Hooks

To reduce the need of back and forth when submitting a pull-request, we use [pre-commit hooks](https://pre-commit.com/).
Once set up, the pre-commit hooks will automatically run every time you make a commit, ensuring code standards are
followed.

To set them up locally, run:

```bash
poetry run pre-commit install
```

### Meta-Grammar

**ArchiTXT** use ANTLR (Another Tool for Language Recognition) to generate a parser/lexer
for the meta-grammar that validates the generated database schemas.
You can view the meta-grammar definition in [`metagrammar.g4`](metagrammar.g4).

If you make changes to the meta-grammar, you’ll need to regenerate the parser/lexer.
To do so, run:

```sh
$ poetry run antlr4 -Dlanguage=Python3 metagrammar.g4 -o architxt/grammar
```

## Pull Request Guidelines

When submitting a pull request, please follow these guidelines:

- **Tests:** Include tests for any new features or bug fixes.
- **Documentation:** Update relevant documentation to reflect your changes.
- **Commit Messages:** Provide clear and concise commit messages using [Gitmoji](https://gitmoji.dev/).

A gitmoji commit message use the format `<intention> [scope?][:?] <message>` where:
   - `intention` is an emoji (either `:shortcode:` or Unicode format) that express the intention of the commit.
   - `scope` is an optional string that adds contextual information.
   - `message` is a brief explanation of the change.

## Pull Request Guidelines

When submitting a pull request, please follow these guidelines:

- **Code Style:** Ensure your code adheres to our coding standards and passes all pre-commit checks.
- **Tests:** Include tests for any new features or bug fixes.
- **Documentation:** Update relevant documentation to reflect your changes.
- **Descriptive Messages:** Provide clear and concise commit messages explaining your changes.

## Need Assistance?

If you have questions or need guidance on contributing, feel free to open an issue or contact the maintainers.
We're here to help and appreciate your interest in improving ArchiTXT!

Thank you for your contributions to ArchiTXT!

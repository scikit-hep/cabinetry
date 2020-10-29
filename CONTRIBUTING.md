# Contributing to cabinetry

Your contributions to `cabinetry` are welcome. Thank you for your interest!

## Issues

[Issues](https://github.com/alexander-held/cabinetry/issues) are a good place to report bugs, ask questions, request features, or discuss potential changes to `cabinetry`.
Before opening a new issue, please have a look through existing issues to avoid duplications.
Please also take a look at the [documentation](https://cabinetry.readthedocs.io/), which may answer some questions.

## Pull requests

It can be helpful to first get into contact via issues before getting started with a [pull request](https://github.com/alexander-held/cabinetry/pulls).
All pull requests are squashed and merged, so feel free to commit as many times as you want to the branch you are working on.
The final commit message should follow the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

## Development environment

For development, install `cabinetry` with the `[develop]` setup extras.
Then install `pre-commit`

```bash
pre-commit install
```

which will run checks before committing any changes.
You can run all tests for `cabinetry` with

```bash
python -m pytest
```

All tests are required to pass before any changes can be merged.
To build the documentation, run

```bash
sphinx-build -W docs docs/_build
```

and open `docs/_build/index.html` in your browser to view the result.

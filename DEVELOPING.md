# Groundlight Tracking Library Developer Guide

The Groundlight Tracking library is a python library that provides various multi-object tracking algorithms on top of the Groundlight Python SDK. This library is designed to easily add object tracking to applications powered by Groundlight.

## Local Development

First, make sure you have the `uv` package manager installed. See installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

### Install dependencies

Then, you can install the package dependencies by running:

```shell
make install

# Or if you want to install the package with extra dev dependencies:
make install-dev
```

### Run tests

You can run the tests by running:

```shell
make test
```

### Run linting

You can run our linting tools by running:

```shell
make lint
```

We're currently using:

- [ruff](https://docs.astral.sh/ruff/) and
  [pylint](https://pylint.readthedocs.io/en/latest/index.html) for general python linting.
- [mypy](https://mypy.readthedocs.io/en/stable/index.html) and
  [ty](https://github.com/astral-sh/ty) for type checking.
- [black](https://black.readthedocs.io/en/stable/index.html) for standardizing code formatting.
- [toml-sort](https://toml-sort.readthedocs.io/en/latest/) for linting the `pyproject.toml` file.

Most of these linters are configured in [pyproject.toml](pyproject.toml) (except for `pylint` in
[.pylintrc](.pylintrc)). Sometimes the linters are wrong or not useful, so we can add overrides. We
prefer to make the smallest possible override:

- single line override with a specific rule (e.g., `# pylint: disable=some-rule` or `# ruff: noqa: F403`)
- single file override with a specific rule
- single file override with a whole class of rules
- global override for a specific rule

### Run formatting

You can run our formatting tools by running:

```shell
make format
```

See [code-quality/format.sh](code-quality/format.sh) for more details on how we format the code.

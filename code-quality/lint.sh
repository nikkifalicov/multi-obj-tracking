#!/usr/bin/env bash
# Lints the codebase.
# Dependencies:
# - uv
# - toml-sort, ruff, pylint, mypy
# Usage:
# $ lint src test bin

if [ $# -eq 0 ]; then
  echo "Usage: $0 TARGET_PATHS"
  exit 1
fi

# Args
TARGET_PATHS="$@"
errors=0

echo "Linting paths: $TARGET_PATHS ..."

echo "-----------------------------------------------------------------------------------------------------------------"

# https://toml-sort.readthedocs.io/en/latest/
echo "Linting with toml-sort (checks pyproject.toml formatting) ..."
uv run toml-sort --check pyproject.toml || { ((errors++)) ; echo "toml-sort found errors that need to be fixed to merge" ;}
echo "-----------------------------------------------------------------------------------------------------------------"

# https://beta.ruff.rs/docs/
echo "Linting with ruff (isort / flake8 / autoflake issues) ..."
uv run ruff check $TARGET_PATHS || { ((errors++)) ; echo "ruff found errors that need to be fixed to merge" ;}
echo "-----------------------------------------------------------------------------------------------------------------"

# https://pylint.readthedocs.io/en/latest/index.html
echo "Linting with pylint ..."
uv run pylint $TARGET_PATHS || { ((errors++)) ; echo "pylint found errors that need to be fixed to merge" ;}
echo "-----------------------------------------------------------------------------------------------------------------"

# https://mypy.readthedocs.io/en/stable/index.html
echo "Linting with mypy (type checking) ..."
uv run mypy $TARGET_PATHS || { ((errors++)) ; echo "mypy found errors that need to be fixed to merge";}
echo "-----------------------------------------------------------------------------------------------------------------"

# https://github.com/astral-sh/ty?tab=readme-ov-file
echo "Linting with ty (type checking) ..."
uv run ty check $TARGET_PATHS || { ((errors++)) ; echo "ty found errors that need to be fixed to merge";}
echo "-----------------------------------------------------------------------------------------------------------------"


if [[ $errors -gt 0 ]]; then
  echo "🚨 linters found $errors errors! "
  exit $errors
fi

echo "✅ Success!"
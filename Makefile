help:  ## Print all targets with their descriptions
	@grep -E '^[a-zA-Z_-]+:.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {if (NF == 1) {printf "\033[36m%-30s\033[0m %s\n", $$1, ""} else {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}}'

# install the package from source
install: 
	uv venv
	uv sync 

# install pre-commit hooks
install-pre-commit: 
	uv run pre-commit install

# install the package from source with extra dev dependencies
# also install pre-commit hooks
install-dev: 
	uv venv
	uv sync --group dev 
	uv run pre-commit install

PYTEST=uv run pytest -v

# You can pass extra arguments to pytest by setting the TEST_ARGS environment variable.
# For example:
# 	`make test TEST_ARGS="-k some_filter"`
TEST_ARGS=

# Record information about the slowest 25 tests (but don't show anything faster than 0.1 seconds)
PROFILING_ARGS = \
	--durations 25 \
	--durations-min 0.1

BASE_TEST_CMD = ${PYTEST} ${PROFILING_ARGS} ${TEST_ARGS}

# Run CPU tests (excludes GPU tests)
test-cpu: install-dev  ## Run CPU tests only (excludes GPU tests)
	${BASE_TEST_CMD} -m "not needsgpu" test

# Run GPU tests only
test-gpu: install-dev  ## Run GPU tests only
	${BASE_TEST_CMD} -m "needsgpu or wantsgpu" test

# Run all tests (CPU and GPU)
test: install-dev  ## Run all tests (CPU and GPU)
	${BASE_TEST_CMD} test

# Run the formatter
format: install-dev
	./code-quality/format.sh src test

# Run linting tools
lint: install-dev
	./code-quality/lint.sh src test

# Run format, lint, and test-cpu - the full validation process for PR readiness without GPU
check-cpu: format lint test-cpu  ## Run format, lint, and CPU tests

# Run format, lint, and test - the full validation process for PR readiness
check: format lint test-cpu test-gpu


# =========================================
# Project configuration
# =========================================
PYTHON      := uv run python
PRECOMMIT   := uv run pre-commit
PYTEST      := uv run pytest

# =========================================
# Help
# =========================================
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install        Sync dependencies & install pre-commit"
	@echo "  make lint           Run linting & formatting"
	@echo "  make typecheck      Run mypy"
	@echo "  make test           Run pytest"
	@echo "  make pre-commit     Run all pre-commit hooks"
	@echo "  make check          Run lint + typecheck + tests"
	@echo "  make clean          Clean caches and artifacts"

# =========================================
# Setup
# =========================================
.PHONY: install
install:
	uv sync
	$(PRECOMMIT) install

# =========================================
# Quality
# =========================================
.PHONY: lint
lint:
	$(PRECOMMIT) run ruff --all-files
	$(PRECOMMIT) run ruff-format --all-files

.PHONY: typecheck
typecheck:
	$(PRECOMMIT) run mypy --all-files

.PHONY: test
test:
	$(PYTEST) --cov=. --cov-report=term-missing


.PHONY: coverage
coverage:
	$(PYTEST) --cov=. --cov-report=html
	@echo "📊 Coverage report generated in htmlcov/index.html"

.PHONY: pre-commit
pre-commit:
	$(PRECOMMIT) run --all-files

.PHONY: check
check: lint typecheck test
	@echo "✅ All checks passed"

# =========================================
# Cleaning
# =========================================
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .cache/pre-commit

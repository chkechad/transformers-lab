# =========================================
# Project configuration
# =========================================
UV          := uv
PYTHON      := $(UV) run python
PRECOMMIT   := $(UV) run pre-commit
PYTEST      := $(UV) run pytest
MKDOCS      := $(UV) run mkdocs
CZ          := $(UV) run cz

SRC := src
TESTS := tests

# =========================================
# Help
# =========================================
.PHONY: help
help:
	@echo "========== Setup =========="
	@echo "make install        Install all dependencies"
	@echo ""
	@echo "========== Quality =========="
	@echo "make lint           Run ruff + format"
	@echo "make typecheck      Run mypy"
	@echo "make test           Run pytest + doctest"
	@echo "make coverage       HTML coverage report"
	@echo "make benchmark      Run performance benchmarks"
	@echo ""
	@echo "========== Docs =========="
	@echo "make doc-serve      Serve docs"
	@echo "make doc-build      Build docs"
	@echo "make doc-deploy     Deploy docs"
	@echo ""
	@echo "========== Release =========="
	@echo "make release        Bump version"
	@echo ""
	@echo "make check          Full extreme pipeline"
	@echo "make clean          Clean project"

# =========================================
# Setup
# =========================================
.PHONY: install
install:
	$(UV) sync --dev --group doc
	$(PRECOMMIT) install
	$(PRECOMMIT) install --hook-type commit-msg

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
	$(PYTEST) tests --cov=transformers_lab --cov-report=term-missing --doctest-modules

.PHONY: coverage
coverage:
	$(PYTEST) --cov=transformers_lab --cov-report=html
	@echo "Coverage report → htmlcov/index.html"

.PHONY: benchmark
benchmark:
	$(PYTEST) --benchmark-only

# =========================================
# Documentation
# =========================================
.PHONY: doc-serve
doc-serve:
	$(MKDOCS) serve

.PHONY: doc-build
doc-build:
	$(MKDOCS) build --strict

.PHONY: doc-deploy
doc-deploy:
	$(MKDOCS) gh-deploy --force

# =========================================
# Release
# =========================================
.PHONY: release
release:
	$(CZ) bump
	@echo "Version bumped + changelog updated"

# =========================================
# Master pipeline
# =========================================
.PHONY: check
check: lint typecheck bandit security test
	@echo "✅✅✅✅"

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
	rm -rf site
	rm -rf dist
	rm -rf build
	rm -rf .cache/pre-commit

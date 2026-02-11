# =========================================
# Project configuration
# =========================================
UV          := uv
PYTHON      := $(UV) run python
PRECOMMIT   := $(UV) run pre-commit
PYTEST      := $(UV) run pytest
MKDOCS      := $(UV) run mkdocs
CZ          := $(UV) run cz
PIPAUDIT    := $(UV) run pip-audit
BANDIT      := $(UV) run bandit
MUTMUT      := $(UV) run mutmut
RADON       := $(UV) run radon
VULTURE     := $(UV) run vulture
CYCLODX     := $(UV) run cyclonedx-py

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
	@echo "========== Security =========="
	@echo "make bandit         Code security scan"
	@echo "make security       Dependency audit"
	@echo "make sbom           Generate SBOM"
	@echo ""
	@echo "========== Advanced Quality =========="
	@echo "make mutation       Run mutation tests"
	@echo "make complexity     Cyclomatic complexity"
	@echo "make deadcode       Detect unused code"
	@echo "make profile        Run profiling"
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
# Security
# =========================================
.PHONY: bandit
bandit:
	$(BANDIT) -q -r $(SRC) -x $(TESTS)

.PHONY: security
security:
	$(PIPAUDIT) --strict

.PHONY: sbom
sbom:
	$(CYCLODX) environment --output-format json --output-file sbom.json
	@echo "SBOM generated → sbom.json"

# =========================================
# Advanced Quality
# =========================================
.PHONY: mutation
mutation:
	$(MUTMUT) run
	$(MUTMUT) results

.PHONY: complexity
complexity:
	$(RADON) cc $(SRC) -a

.PHONY: deadcode
deadcode:
	$(VULTURE) $(SRC)

.PHONY: profile
profile:
	$(PYTHON) -m cProfile -o profile.out -m pytest
	@echo "Profile saved → profile.out"

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
	rm -rf .mutmut-cache
	rm -rf profile.out
	rm -rf sbom.json
	rm -rf .cache/pre-commit

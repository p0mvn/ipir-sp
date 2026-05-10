# InspiRING.Pack project root Makefile.
#
# Targets dispatch to subprojects under tools/ and (later) the Rust crate.
# Run `make help` for the full list. Conventions:
#
#   <ns>-sync          install/update dependencies for namespace <ns>
#   <ns>-test          run all tests for namespace <ns>
#   <ns>-test-stage    run a single stage's tests (uses STAGE=<name>)
#   <ns>-clean         remove build artifacts for namespace <ns>
#
# Namespaces:
#   oracle             Python reference oracle at tools/python-oracle/
#   crate              (added in Phase 4) Rust crate

ORACLE_DIR := tools/python-oracle

FIXTURES_DIR := fixtures

.DEFAULT_GOAL := help
.PHONY: help \
        oracle-sync oracle-test oracle-test-stage oracle-test-fast oracle-clean \
        oracle-fixtures \
        test sync clean

help: ## Show this help
	@awk 'BEGIN { FS = ":.*##"; printf "\n%-22s %s\n", "Target", "Description"; printf "%-22s %s\n", "------", "-----------" } \
	      /^[a-zA-Z_-]+:.*##/ { printf "%-22s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nExamples:\n"
	@printf "  make oracle-test                          # run all oracle tests\n"
	@printf "  make oracle-test-stage STAGE=params       # run Stage 0 only\n"
	@printf "  make oracle-test-stage STAGE=ring         # run Stage 1 only (once added)\n\n"

# ---- Aggregate targets (operate across all namespaces) ----

sync: oracle-sync ## Install/update all dependencies

test: oracle-test ## Run all tests across all namespaces

clean: oracle-clean ## Remove all build/test artifacts

# ---- Python oracle (tools/python-oracle/) ----

oracle-sync: ## Install/update Python oracle dependencies via uv
	cd $(ORACLE_DIR) && uv sync

oracle-test: oracle-sync ## Run the full Python oracle pytest suite (verbose)
	cd $(ORACLE_DIR) && uv run pytest -v

oracle-test-fast: oracle-sync ## Run the full Python oracle pytest suite (quiet)
	cd $(ORACLE_DIR) && uv run pytest -q

oracle-test-stage: oracle-sync ## Run one stage's tests; usage: make oracle-test-stage STAGE=<name>
	@if [ -z "$(STAGE)" ]; then \
	  echo "Usage: make oracle-test-stage STAGE=<name>"; \
	  echo "Available stages:"; \
	  ls $(ORACLE_DIR)/tests/test_*.py 2>/dev/null | sed 's|.*/test_||;s|\.py$$||' | awk '{ printf "  - %s\n", $$0 }'; \
	  exit 1; \
	fi
	cd $(ORACLE_DIR) && uv run pytest -v tests/test_$(STAGE).py

oracle-fixtures: oracle-sync ## Regenerate JSON fixtures into ./fixtures/ for the Rust crate
	cd $(ORACLE_DIR) && uv run python scripts/generate_fixtures.py --output ../../$(FIXTURES_DIR)

oracle-clean: ## Remove Python oracle build/test artifacts
	rm -rf $(ORACLE_DIR)/.pytest_cache $(ORACLE_DIR)/.hypothesis
	find $(ORACLE_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(ORACLE_DIR) -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# Convenience targets for the laptop / Gymnasium path. These wrap the
# exact commands documented in README.md; nothing here is required to use
# the package. Run from the repository root: `make <target>`.

.PHONY: help install dev cache check test test-fast smoke clean

help:  ## List the available targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Editable install of the wrapper + Gymnasium path (no JAX).
	pip install -e .

dev:  ## Editable install with the dev extras (pytest, xdist, gdown).
	pip install -e ".[dev]"

cache:  ## Cache the Phase-4 OFFLINE game files (needs ARC_API_KEY).
	python scripts/cache_env_files.py

check:  ## One-command install sanity check (no network, no game files).
	python -m arc3_wm

test:  ## Run the full test suite.
	pytest

test-fast:  ## Run the suite in parallel (needs pytest-xdist).
	pytest -n auto

smoke:  ## Random agent on vc33 for 3 episodes (needs cached env files).
	python examples/random_agent.py --game vc33 --episodes 3

clean:  ## Remove Python caches and build artifacts.
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('*.egg-info')]"

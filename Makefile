.PHONY: install native test unit e2e bench clean

install:
	python3 -m pip install -e ".[dev]"

native:
	python3 setup.py build_ext --inplace

unit:
	python3 -m pytest -q -m "not e2e"

e2e:
	python3 -m pytest -q -m e2e

test:
	python3 -m pytest -q

bench:
	python3 benchmarks/benchmark.py --help

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

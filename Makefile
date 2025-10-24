PY?=python
VENV=.venv

.PHONY: help install figures clean

help:
	@echo "Targets:"
	@echo "  install   - create venv and install requirements"
	@echo "  figures   - generate all PDFs into ./figures/"
	@echo "  clean     - remove cache/aux files"

install:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

figures:
	$(PY) make_figures.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type f -name '*.aux' -delete
	find . -type f -name '*.log' -delete
	find . -type f -name '*.synctex*' -delete

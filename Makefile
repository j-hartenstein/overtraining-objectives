PYTHON ?= python3

.PHONY: reproduce verify

reproduce:
	$(PYTHON) run_all.py

verify:
	$(PYTHON) verify.py

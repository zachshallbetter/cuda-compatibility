PYTHON ?= python3
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: check test context

context:
	$(PYTHON) $(PROJECT_ROOT)/scripts/gen-context.py

check:
	$(PYTHON) $(PROJECT_ROOT)/scripts/validate-bootstrap.py --check-context
	$(PYTHON) $(PROJECT_ROOT)/scripts/check-protocol-integrity.py
	$(PYTHON) $(PROJECT_ROOT)/scripts/site_checks.py

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PROJECT_ROOT)/src $(PYTHON) -m unittest discover -s $(PROJECT_ROOT)/tests -v

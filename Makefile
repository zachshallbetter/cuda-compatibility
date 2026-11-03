PYTHON ?= python3
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: check test context site cards live-audit

context:
	$(PYTHON) $(PROJECT_ROOT)/.agents/bootstrap/scripts/gen-context.py

check:
	$(PYTHON) $(PROJECT_ROOT)/.agents/bootstrap/scripts/validate-bootstrap.py --check-context
	$(PYTHON) $(PROJECT_ROOT)/.agents/bootstrap/scripts/check-protocol-integrity.py
	$(PYTHON) $(PROJECT_ROOT)/scripts/site_checks.py
	$(PYTHON) $(PROJECT_ROOT)/scripts/build_site_meta.py --check

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PROJECT_ROOT)/src $(PYTHON) -m unittest discover -s $(PROJECT_ROOT)/tests -v

# Regenerate everything derived from site/ pages, then gate it (docs/SITE_PROTOCOL.md, step 3).
site:
	$(PYTHON) $(PROJECT_ROOT)/scripts/build_site_meta.py
	$(PYTHON) $(PROJECT_ROOT)/scripts/build_llms_full.py
	$(MAKE) context check test

# Share cards and the no-JavaScript wireframe; needs Google Chrome and Pillow.
cards:
	$(PYTHON) $(PROJECT_ROOT)/scripts/build_social_cards.py

# After a merge has deployed: the public site must serve this checkout exactly.
live-audit:
	$(PYTHON) $(PROJECT_ROOT)/scripts/site_live_audit.py

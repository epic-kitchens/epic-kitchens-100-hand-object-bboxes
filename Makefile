.PHONY: docs
docs:
	$(MAKE) -C docs html

.PHONY: test
test:
	python -m pytest tests

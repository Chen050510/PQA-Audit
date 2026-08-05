.PHONY: verify clean

verify:
	python3 scripts/verify_code.py
	python3 scripts/verify_results.py

clean:
	find scripts software -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .cache

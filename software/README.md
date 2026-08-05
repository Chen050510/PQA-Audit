# Analysis Software

This directory contains prompt contracts, final-answer readers, runtime helpers,
and offline analysis programs for paired multiple-choice outputs.

## Components

- `recurrence/`: compares independent launches and related Process phrasings;
- `cross_dataset/`: computes paired gains and losses across datasets;
- `reproduce_key_results.py`: recomputes selected paired summaries from a
  prepared analysis-data directory.

The repository does not include generated outputs or benchmark data. Supply the
required run directory through each command-line entry point. From the repository
root, run `make verify` to check source hashes, Python syntax, JSON configurations,
and CLI availability.

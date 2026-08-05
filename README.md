# Direction 4: Paired-Question Analysis

This repository contains code for studying how response instructions change
which multiple-choice questions a language model answers correctly.

The analysis pairs responses to the same question before aggregation. It then
separates questions that become correct from questions that become wrong,
checks whether related instruction phrasings affect some of the same questions,
and repeats the paired comparison across datasets.

## Repository layout

```text
software/
  recurrence/              Recurrence analysis across related phrasings
  cross_dataset/           Paired analysis across supporting datasets
  reproduce_key_results.py Reproduce selected summaries from prepared outputs
data/
  anonymous_data.zip       Sanitized generated outputs and frozen summaries
  validation.json          Validation record for the sanitized archive
scripts/
  verify_code.py           Source, configuration, and CLI validation
  verify_results.py        Checksum and offline result reproduction
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Validate the code

```bash
make verify
```

The validation is CPU-only. It verifies the source and data checksums, compiles
every Python file, parses every JSON configuration, checks the command-line entry
points, and reproduces selected paired summaries from the sanitized outputs. It
does not download datasets, load model weights, or generate answers.

## Run the analyses

The analysis programs accept prepared per-question model outputs. The included
sanitized archive provides the outputs needed by `make verify`; benchmark
question text and choices are not redistributed.

```bash
python software/recurrence/scripts/analyze_e31.py \
  --run-dir /path/to/recurrence_run \
  --output-dir /path/to/recurrence_analysis

python software/cross_dataset/scripts/analyze_e32.py \
  --run-dir /path/to/cross_dataset_run \
  --output-dir /path/to/cross_dataset_analysis
```

See [docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md) for the included components and
their boundaries.

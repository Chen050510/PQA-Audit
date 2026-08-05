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
  reproduce_key_results.py Optional analysis entry point for prepared outputs
scripts/
  verify_code.py           Source, configuration, and CLI validation
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

The validation is CPU-only. It verifies the source manifest, compiles every
Python file, parses every JSON configuration, and checks the command-line entry
points. It does not download datasets, load model weights, or generate answers.

## Run the analyses

The analysis programs expect prepared per-question model outputs. Data is not
included in this repository.

```bash
python software/recurrence/scripts/analyze_e31.py \
  --run-dir /path/to/recurrence_run \
  --output-dir /path/to/recurrence_analysis

python software/cross_dataset/scripts/analyze_e32.py \
  --run-dir /path/to/cross_dataset_run \
  --output-dir /path/to/cross_dataset_analysis
```

See [docs/CODE_SCOPE.md](docs/CODE_SCOPE.md) for the included components and
their boundaries.

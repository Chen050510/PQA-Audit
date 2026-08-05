# Code Scope

## Included

- readers for submitted final answers;
- prompt and runtime helpers;
- record and dataset utilities;
- recurrence analysis across related instruction phrasings;
- paired gain/loss analysis across datasets;
- frozen JSON configurations;
- sanitized generated outputs and frozen summaries;
- source, command-line, checksum, and result validation.

## Not included

- benchmark question text or dataset copies;
- model weights or tokenizer files;
- cloud execution logs or private machine paths.

The repository distributes analysis code and the sanitized outputs needed to
reproduce selected summaries. Running a new comparison requires separately
prepared per-question outputs that match the expected schemas.

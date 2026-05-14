# fineweb-edu-pretraining-lab

> Reproducible GPT pretraining pipeline on the FineWeb-Edu dataset.

![Status](https://img.shields.io/badge/status-WIP-orange)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Project goals

- Build a small-but-honest GPT pretraining pipeline end-to-end on FineWeb-Edu, with reproducibility as a first-class concern (deterministic data sharding, seed control, config-driven runs).
- Stay within a single-GPU budget (RTX 3060 12GB) while exercising the same engineering shape as a multi-node setup: sharded preprocessing, AMP, gradient accumulation, checkpoint/resume, W&B logging.
- Treat the repo as a portfolio artifact — clean abstractions, type-checked modules, tests for the data and model layers, no notebook-only workflows.
- Document trade-offs (what was simplified for the 3060 budget, what would change at scale) so the project reads as engineering, not a tutorial.

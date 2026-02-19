# Engine Experiment Harness

This repository implements a stage-gated, equal-cost experiment program for six model-engine tracks:

1. Hybrid photonic-digital attention acceleration (`T1`)
2. Reversible-state Transformer blocks (`T2`)
3. Compression-first hierarchical memory (`T3`)
4. Vector-symbolic scratchpad reasoning (`T4`)
5. Self-assembling modular circuits (`T5`)
6. Energy-based global decoding (`T6`)

A shared permanent `ANCHOR` baseline engine is included in every stage and used for cross-window stable comparisons.

## CLI

After installation (`pip install -e .`), the CLI is:

```bash
exp run --spec specs/stage1/t1_e1.yaml
exp eval --run <run_id>
exp compare --candidate <run_id> --baseline <run_id>
exp gate --stage 1
exp memo --stage 1
exp gate --stage 4 --marker artifacts/memos/full_program_run_marker.txt
```

Artifacts are written to:

- `artifacts/runs/*.json`
- `artifacts/comparisons/*.json`
- `artifacts/memos/*.md`

## Additional CLI Commands

In addition to the commands listed above, the following commands are also available:

- **Generate experiment specifications**:
  ```bash
  exp generate --stage <stage_number> --type <main|recovery>
  ```
  Generate experiment specifications for a specific stage. Use the `--type` flag to specify the type of specs to generate (default: `main`).

- **Package the project for PyPI publishing**:
  ```bash
  exp package --version <version_number> --output <output_directory>
  ```
  Package the project for publishing to PyPI. Specify the version number and optionally the output directory (default: `dist/`).

- **Run the test suite**:
  ```bash
  exp test --ci
  ```
  Run the test suite. Use the `--ci` flag to run tests in CI mode.

- **Publish the package to PyPI**:
  ```bash
  exp publish --repository <repository_name>
  ```
  Publish the package to a PyPI repository (default: `pypi`).

## Spec Catalog

Generate or refresh all stage/track specs:

```bash
python scripts/generate_specs.py
```

Batch-run a full stage (all tracks, baseline + experiments):

```bash
python scripts/run_stage.py --stage 1
python scripts/run_stage.py --stage 1 --all-seeds
python scripts/run_stage.py --stage 2 --all-seeds --selection-marker artifacts/memos/full_program_run_marker.txt
python scripts/run_stage.py --stage 3 --all-seeds --selection-marker artifacts/memos/full_program_run_marker.txt
python scripts/run_stage.py --stage 4 --all-seeds
```

`run_stage.py` automatically runs `specs/stageX/anchor_baseline.yaml` once per seed and records anchor-relative deltas in comparison reports.
For stage 2 and stage 3, it auto-selects promoted tracks from the previous stage gate output (override with `--tracks T3,T4` when needed).

Run focused recovery sweeps for T4/T5:

```bash
python scripts/run_recovery.py --stage 2 --track all --all-seeds
python scripts/run_recovery.py --stage 3 --track T5 --all-seeds
```

Recovery runs compare candidates against a **cost-matched baseline** (same budget/context/seed) to enforce the equal-cost rule fairly.

Create a clean run window + consolidated memo + manifest:

```bash
python scripts/start_run_window.py
# run stage batches/recovery commands...
python scripts/build_consolidated_memo.py
```

This writes:

- `artifacts/memos/consolidated_final_ranking.md`
- `artifacts/memos/final_run_manifest.json`

The consolidated memo ranks tracks primarily by anchor-relative gains.
Final ordering uses a pass-adjusted decision score and excludes recovery sweeps from primary stage ranking stats.

## Protocol Defaults

- Equal-cost comparison tolerance: `+/-2%`
- Composite score: `0.45 * long_context + 0.35 * reasoning + 0.20 * consistency`
- Promotion gates:
  - Stage 1 -> 2: `delta_composite >= 3`, stable training, fluency drop <= 2%
  - Stage 2 -> 3: `delta_composite >= 5`, latency overhead <= 15%
  - Stage 3 final: `delta_composite >= 8`, bootstrap CI excludes 0
  - Stage 4 (T3 confirmatory): `delta_composite >= 8`, bootstrap CI excludes 0, T3-specific pass required

### Additional Details

- **SEED_POLICY**: Defines the minimum number of seeds required for each stage. For example:
  - Stage 1: Minimum 3 seeds
  - Stage 2: Minimum 5 seeds
  - Stage 3: Minimum 8 seeds

- **STAGE_BUDGET_GPU_HOURS**: Specifies the maximum GPU hours allowed per stage. Each stage has a predefined budget to ensure fair comparisons.

## Schema Validation

The Engine Experiment Harness uses schema validation to ensure that experiment specifications and results conform to defined standards. The following schemas are used:

- **`experiment_spec`**: Validates the structure and content of experiment specifications.
- **`run_result`**: Ensures that run results are correctly formatted and include all required fields.
- **`comparison_report`**: Validates the structure of comparison reports.

These schemas are located in the `schemas/` directory and are loaded and validated using the `validate_schema` and `load_schema` functions in the codebase.

## Stage-Gated Experimentation

The stage-gated experimentation process is implemented to ensure a structured and fair evaluation of experimental tracks. The following functions are used to manage tracks and stages:

- **Track Discovery**: The `_discover_tracks` function in `scripts/run_stage.py` identifies all tracks for a given stage by scanning the `specs/stageX/` directory for baseline YAML files.
- **Track Overrides**: The `_parse_track_overrides` function in `scripts/run_stage.py` allows for specifying custom tracks to override the default selection.

This process ensures that only the most promising tracks are promoted to the next stage based on the defined gating criteria.

## Overview

The Engine Experiment Harness is a modular and reusable framework for running and managing stage-gated, equal-cost experiment programs. It is designed to facilitate experimentation across multiple model-engine tracks, providing tools for running experiments, evaluating results, and generating reports. The framework is built with modularity and reusability in mind, making it easy to extend and adapt for new experiments.

## Features

- **Stage-Gated Experimentation**: Supports multi-stage experiments with clear gating criteria.
- **Equal-Cost Comparisons**: Ensures fair comparisons between experimental and baseline models.
- **Modular Design**: Easily extendable to support new experiment tracks and stages.
- **Comprehensive CLI**: Simplifies running experiments, evaluations, comparisons, and report generation.
- **Schema Validation**: Ensures experiment specifications and results conform to defined schemas.
- **Artifact Management**: Automatically organizes and stores experiment artifacts, comparisons, and memos.

## Installation

To install the Engine Experiment Harness, clone the repository and install the dependencies:

```bash
git clone https://github.com/toxzak-svg/engine.git
cd engine
pip install -e .
```

## Usage

The CLI provides the following commands for managing experiments:

- **Run an experiment**:
  ```bash
  exp run --spec <path_to_spec_file>
  ```
- **Evaluate a run**:
  ```bash
  exp eval --run <run_id>
  ```
- **Compare results**:
  ```bash
  exp compare --candidate <run_id> --baseline <run_id>
  ```
- **Gate experiments**:
  ```bash
  exp gate --stage <stage_number>
  ```
- **Generate memos**:
  ```bash
  exp memo --stage <stage_number>
  ```

## Contributing

We welcome contributions to the Engine Experiment Harness! To contribute:

1. Fork the repository and create a new branch for your feature or bugfix.
2. Write clear and concise code, following the existing style and conventions.
3. Add tests for your changes in the `tests/` directory.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

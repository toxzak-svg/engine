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

## Spec Catalog

Generate or refresh all stage/track specs:

```bash
python scripts/generate_specs.py
```

Batch-run a full stage (all tracks, baseline + experiments):

```bash
python scripts/run_stage.py --stage 1
python scripts/run_stage.py --stage 1 --all-seeds
python scripts/run_stage.py --stage 4 --all-seeds
```

`run_stage.py` automatically runs `specs/stageX/anchor_baseline.yaml` once per seed and records anchor-relative deltas in comparison reports.

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

## Protocol Defaults

- Equal-cost comparison tolerance: `+/-2%`
- Composite score: `0.45 * long_context + 0.35 * reasoning + 0.20 * consistency`
- Promotion gates:
  - Stage 1 -> 2: `delta_composite >= 3`, stable training, fluency drop <= 2%
  - Stage 2 -> 3: `delta_composite >= 5`, latency overhead <= 15%
  - Stage 3 final: `delta_composite >= 8`, bootstrap CI excludes 0
  - Stage 4 (T3 confirmatory): `delta_composite >= 8`, bootstrap CI excludes 0, T3-specific pass required

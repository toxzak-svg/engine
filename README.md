# Engine Experiment Harness

This repository implements a stage-gated, equal-cost experiment program for six model-engine tracks:

1. Hybrid photonic-digital attention acceleration (`T1`)
2. Reversible-state Transformer blocks (`T2`)
3. Compression-first hierarchical memory (`T3`)
4. Vector-symbolic scratchpad reasoning (`T4`)
5. Self-assembling modular circuits (`T5`)
6. Energy-based global decoding (`T6`)

## CLI

After installation (`pip install -e .`), the CLI is:

```bash
exp run --spec specs/stage1/t1_e1.yaml
exp eval --run <run_id>
exp compare --candidate <run_id> --baseline <run_id>
exp gate --stage 1
exp memo --stage 1
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
```

## Protocol Defaults

- Equal-cost comparison tolerance: `+/-2%`
- Composite score: `0.45 * long_context + 0.35 * reasoning + 0.20 * consistency`
- Promotion gates:
  - Stage 1 -> 2: `delta_composite >= 3`, stable training, fluency drop <= 2%
  - Stage 2 -> 3: `delta_composite >= 5`, latency overhead <= 15%
  - Stage 3 final: `delta_composite >= 8`, bootstrap CI excludes 0

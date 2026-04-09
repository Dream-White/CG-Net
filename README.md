# Traffic GNN Paper Project

This repository is a modularized implementation split from the original monolithic `GNN0.py`, prepared for public release and paper reproducibility.

## Highlights

- Dynamic graph learning without fixed topology priors
- Physics-aware weather fusion for robust forecasting
- Probabilistic output (`mu`, `log_var`) with hybrid physics regularization
- Publication-ready plotting pipeline (including 3-case comparison figure)

## Project Structure

```text
traffic-gnn-paper/
  configs/
    default.json
  scripts/
    run_paper.py
    run_from_config.py
    smoke_test.py
  src/
    traffic_gnn/
      baselines.py
      config.py
      data.py
      losses.py
      models.py
      pipeline.py
      plotting.py
      compat.py
  REPRODUCIBILITY.md
  requirements.txt
  pyproject.toml
```

## Installation

### Option A: lightweight install

```bash
pip install -r requirements.txt
```

### Option B: editable package install

```bash
pip install -e .
```

## Quick Start

```bash
python scripts/run_paper.py --data_path "path/to/pems04_fused.npz" --output_dir outputs --epochs 100 --batch_size 64 --seed 42
```

Run from JSON config:

```bash
python scripts/run_from_config.py --config configs/default.json --data_path "path/to/pems04_fused.npz"
```

Fast smoke test:

```bash
python scripts/smoke_test.py --data_path "path/to/pems04_fused.npz"
```

## Main Outputs

- `outputs/best_model.pth`
- `outputs/fig_training_curve.png`
- `outputs/fig_three_case_curves_topjournal.png`

## Data

Supported input formats:

- `.npz` (preferred, with `data` key)
- `.h5`
- `.csv`

## Reproducibility

See `REPRODUCIBILITY.md` for environment, run commands, and reproduction notes.

## Citation

Please cite your related paper and this repository.
See `CITATION.cff`.

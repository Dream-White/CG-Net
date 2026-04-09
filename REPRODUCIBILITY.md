# Reproducibility Guide

This document describes the minimum steps required to reproduce the core paper artifacts.

## 1) Environment

- Python: 3.10+ recommended
- OS: Windows/Linux
- GPU: optional (CUDA-capable GPU recommended for full training)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Data

Provide a dataset file path (`.npz`, `.h5`, or `.csv`) with shape compatible to the loader:

- Preferred: `data` key in `.npz`, shape `[T, N, F]`
- Auto-adapt is supported for `[T, N]` and lower feature dims

Example path:

```text
data/pems04_fused.npz
```

## 3) Run Main Pipeline

```bash
python scripts/run_paper.py --data_path "data/pems04_fused.npz" --output_dir outputs --epochs 100 --batch_size 64 --seed 42
```

## 4) Smoke Test (fast)

Use a very short run to verify environment and code path:

```bash
python scripts/smoke_test.py --data_path "data/pems04_fused.npz"
```

## 5) Expected Core Outputs

- `outputs/best_model.pth`
- `outputs/fig_training_curve.png`
- `outputs/fig_three_case_curves_topjournal.png`

## 6) Reproducibility Notes

- Randomness is controlled by `seed` via `set_random_seed()`.
- Report paper metrics with multiple seeds (e.g., 42/43/44) and aggregate mean/std in manuscript tables.
- Keep dataset split and preprocessing identical across model comparisons.

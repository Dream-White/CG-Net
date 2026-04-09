import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_gnn.config import ExperimentConfig
from traffic_gnn.pipeline import run_train_pipeline
from traffic_gnn.plotting import plot_three_case_curves_hd_png


def main():
    parser = argparse.ArgumentParser(description="Run CG-Uncertainty-Net paper pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to npz/h5/csv dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = run_train_pipeline(cfg)

    # Use ours output as placeholders for baseline arrays if user has not yet migrated baseline modules.
    plot_three_case_curves_hd_png(
        result["pred_gru"],
        result["pred_static"],
        result["pred_ours"],
        result["true_ours"],
        result["scaler_mean"],
        result["scaler_std"],
        save_path=f"{cfg.output_dir}/fig_three_case_curves_topjournal.png",
        sample_idx=50,
    )
    print("Done. Artifacts saved under:", cfg.output_dir)


if __name__ == "__main__":
    main()

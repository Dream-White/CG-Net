import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run paper pipeline from JSON config")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config json")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if args.data_path is not None:
        raw["data_path"] = args.data_path
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir

    cfg = ExperimentConfig(**raw)
    result = run_train_pipeline(cfg)

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

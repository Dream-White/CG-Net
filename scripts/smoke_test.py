import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_gnn.config import ExperimentConfig
from traffic_gnn.pipeline import run_train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Fast smoke test for pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_path=args.data_path,
        output_dir="outputs/smoke_test",
        epochs=2,
        batch_size=16,
        seed=123,
    )
    run_train_pipeline(cfg)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

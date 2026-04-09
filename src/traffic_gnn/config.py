from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch


@dataclass
class ExperimentConfig:
    data_path: str
    output_dir: str = "outputs"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.002
    input_window: int = 12
    output_window: int = 12
    traffic_dim: int = 3
    weather_dim: int = 4
    hidden_dim: int = 64
    node_emb_dim: int = 10
    time_emb_dim: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_output_dir(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

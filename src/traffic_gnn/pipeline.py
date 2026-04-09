import torch

from .config import ExperimentConfig, set_random_seed
from .data import create_dataloaders
from .models import CGUncertaintyNet
from .train import train_probabilistic_model, collect_predictions
from .baselines import GRUBaseline, StaticGCNBaseline, train_baseline
from .plotting import setup_plot_style, plot_training_curve


def run_train_pipeline(cfg: ExperimentConfig):
    set_random_seed(cfg.seed)
    cfg.ensure_output_dir()
    setup_plot_style()

    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        cfg.data_path, cfg.input_window, cfg.output_window, cfg.batch_size
    )
    cfg.num_nodes = int(train_ds.data.shape[1])

    model = CGUncertaintyNet(cfg).to(cfg.device)
    history, ckpt_path = train_probabilistic_model(model, train_loader, val_loader, cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))

    gru_model = GRUBaseline(cfg).to(cfg.device)
    static_model = StaticGCNBaseline(cfg).to(cfg.device)
    pred_gru, true_gru = train_baseline(gru_model, "GRU-Base", train_loader, test_loader, cfg)
    pred_static, _ = train_baseline(static_model, "Static-GCN", train_loader, test_loader, cfg)
    pred_ours, true_ours = collect_predictions(model, test_loader, cfg.device)
    plot_training_curve(history, f"{cfg.output_dir}/fig_training_curve.png")

    return {
        "model": model,
        "history": history,
        "pred_ours": pred_ours,
        "true_ours": true_ours,
        "pred_gru": pred_gru,
        "true_gru": true_gru,
        "pred_static": pred_static,
        "test_loader": test_loader,
        "scaler_mean": train_ds.mean,
        "scaler_std": train_ds.std,
    }

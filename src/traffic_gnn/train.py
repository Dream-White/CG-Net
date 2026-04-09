from pathlib import Path
import numpy as np
import torch

from .losses import HybridPhysicsLoss


def train_probabilistic_model(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = HybridPhysicsLoss(alpha=0.1, beta=0.01, gamma=0.01, weather_mod=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    ckpt_path = Path(cfg.output_dir) / "best_model.pth"

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for x, y, t_idx in train_loader:
            x, y, t_idx = x.to(cfg.device), y.to(cfg.device), t_idx.to(cfg.device)
            optimizer.zero_grad()
            mu, log_var = model(x, t_idx)
            loss = criterion(mu, log_var, y, x_context=x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, t_idx in val_loader:
                x, y, t_idx = x.to(cfg.device), y.to(cfg.device), t_idx.to(cfg.device)
                mu, log_var = model(x, t_idx)
                loss = criterion(mu, log_var, y, x_context=x)
                val_losses.append(loss.item())

        tr_loss = float(np.mean(train_losses))
        va_loss = float(np.mean(val_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{cfg.epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

    return history, ckpt_path


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, t_idx in loader:
        x, y, t_idx = x.to(device), y.to(device), t_idx.to(device)
        mu, _ = model(x, t_idx)
        preds.append(mu.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

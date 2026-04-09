from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True,
        }
    )


def plot_training_curve(history: dict, save_path: str):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss (NLL+FFT)", color="#1f77b4", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss (NLL+FFT)", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Training Convergence Analysis", fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_three_case_curves_hd_png(pred_gru, pred_static, pred_ours, true_data, scaler_mean, scaler_std,
                                  save_path="outputs/fig_three_case_curves_topjournal.png",
                                  sample_idx=50, node_ids=None,
                                  y_share="global",
                                  legend_mode="top",
                                  title_mode="compact"):
    for name, arr in [("pred_gru", pred_gru), ("pred_static", pred_static), ("pred_ours", pred_ours), ("true_data", true_data)]:
        if not hasattr(arr, "shape") or len(arr.shape) != 3:
            raise ValueError(f"{name} must have shape (N, H, V), got {getattr(arr, 'shape', None)}")
    if pred_gru.shape != true_data.shape or pred_static.shape != true_data.shape or pred_ours.shape != true_data.shape:
        raise ValueError("Prediction/target shapes are inconsistent.")

    flow_mean = scaler_mean.flatten()[0] if hasattr(scaler_mean, "flatten") else scaler_mean
    flow_std = scaler_std.flatten()[0] if hasattr(scaler_std, "flatten") else scaler_std

    t_real = true_data * flow_std + flow_mean
    p_gru = pred_gru * flow_std + flow_mean
    p_static = pred_static * flow_std + flow_mean
    p_ours = pred_ours * flow_std + flow_mean

    n_samples, horizon, n_nodes = t_real.shape
    safe_sample_idx = int(np.clip(sample_idx, 0, n_samples - 1))
    order = np.argsort(np.std(t_real.reshape(-1, n_nodes), axis=0))

    if node_ids is None:
        node_ids = [
            int(order[max(0, int(0.15 * (n_nodes - 1)))]),
            int(order[max(0, int(0.50 * (n_nodes - 1)))]),
            int(order[max(0, int(0.85 * (n_nodes - 1)))]),
        ]
    else:
        node_ids = [int(np.clip(i, 0, n_nodes - 1)) for i in list(node_ids)[:3]]
        while len(node_ids) < 3:
            node_ids.append(node_ids[-1] if node_ids else 0)

    unique_ids = []
    for nid in node_ids:
        if nid not in unique_ids:
            unique_ids.append(nid)
    for cand in order.tolist():
        if len(unique_ids) == 3:
            break
        if cand not in unique_ids:
            unique_ids.append(int(cand))
    node_ids = unique_ids[:3]

    c_true = "#1f77b4"
    c_gru = "#00A087"
    c_static = "#4DBBD5"
    c_ours = "#E64B35"

    x = np.arange(1, horizon + 1)
    if str(y_share).lower() == "global":
        y_pool = np.concatenate(
            [
                t_real[safe_sample_idx, :, node_ids].reshape(-1),
                p_gru[safe_sample_idx, :, node_ids].reshape(-1),
                p_static[safe_sample_idx, :, node_ids].reshape(-1),
                p_ours[safe_sample_idx, :, node_ids].reshape(-1),
            ]
        )
        y_min = float(np.nanmin(y_pool))
        y_max = float(np.nanmax(y_pool))
        pad = 0.06 * (y_max - y_min + 1e-6)
        y_lim = (y_min - pad, y_max + pad)
    else:
        y_lim = None

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12.0), sharex=True)
    letters = ["(a)", "(b)", "(c)"]
    style_true = dict(color=c_true, linewidth=2.2, linestyle="-")
    style_gru = dict(color=c_gru, linewidth=1.8, linestyle="--", alpha=0.95)
    style_static = dict(color=c_static, linewidth=1.8, linestyle=":", alpha=0.95)
    style_ours = dict(color=c_ours, linewidth=2.4, linestyle="-")

    for i, node_id in enumerate(node_ids):
        ax = axes[i]
        ax.plot(x, t_real[safe_sample_idx, :, node_id], label="Raw cellular data", **style_true, zorder=4)
        ax.plot(x, p_gru[safe_sample_idx, :, node_id], label="GRU", **style_gru, zorder=2)
        ax.plot(x, p_static[safe_sample_idx, :, node_id], label="Static-GCN", **style_static, zorder=2)
        ax.plot(x, p_ours[safe_sample_idx, :, node_id], label="CG-Net (Ours)", **style_ours, zorder=3)
        ax.set_ylabel("The number of cellular traffic CDRs", fontsize=11)
        ax.grid(True, which="major", alpha=0.25, linestyle="--")
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.10, linestyle=":")
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.set_title(f"Node {node_id}" if str(title_mode).lower() != "full" else f"Node {node_id} case (sample={safe_sample_idx})", fontsize=11, pad=6)
        ax.text(0.5, -0.24, letters[i], transform=ax.transAxes, ha="center", va="top", fontsize=12)
        if str(legend_mode).lower() == "each":
            ax.legend(loc="upper left", fontsize=8, frameon=True)

    if str(legend_mode).lower() == "top":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, fontsize=9)

    axes[-1].set_xlabel("Time points", fontsize=12)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98] if str(legend_mode).lower() == "top" else None)
    fig.savefig(str(out), dpi=600, bbox_inches="tight")
    plt.close(fig)


def export_all_nodes_case_curves(
    pred_gru, pred_static, pred_ours, true_data, scaler_mean, scaler_std,
    out_dir="outputs/all_nodes_cases",
    sample_idx=50,
    per_page=16,
    max_pages=None,
):
    for name, arr in [("pred_gru", pred_gru), ("pred_static", pred_static), ("pred_ours", pred_ours), ("true_data", true_data)]:
        if not hasattr(arr, "shape") or len(arr.shape) != 3:
            raise ValueError(f"{name} must have shape (N, H, V).")
    if pred_gru.shape != true_data.shape or pred_static.shape != true_data.shape or pred_ours.shape != true_data.shape:
        raise ValueError("Prediction/target shapes are inconsistent.")

    flow_mean = scaler_mean.flatten()[0] if hasattr(scaler_mean, "flatten") else scaler_mean
    flow_std = scaler_std.flatten()[0] if hasattr(scaler_std, "flatten") else scaler_std

    t_real = true_data * flow_std + flow_mean
    p_gru = pred_gru * flow_std + flow_mean
    p_static = pred_static * flow_std + flow_mean
    p_ours = pred_ours * flow_std + flow_mean

    n_samples, horizon, n_nodes = t_real.shape
    safe_sample_idx = int(np.clip(sample_idx, 0, n_samples - 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_true = "#1f77b4"
    c_gru = "#00A087"
    c_static = "#4DBBD5"
    c_ours = "#E64B35"
    style_true = dict(color=c_true, linewidth=1.6, linestyle="-")
    style_gru = dict(color=c_gru, linewidth=1.2, linestyle="--", alpha=0.95)
    style_static = dict(color=c_static, linewidth=1.2, linestyle=":", alpha=0.95)
    style_ours = dict(color=c_ours, linewidth=1.6, linestyle="-")

    x = np.arange(1, horizon + 1)
    y_pool = np.concatenate(
        [
            t_real[safe_sample_idx, :, :].reshape(-1),
            p_gru[safe_sample_idx, :, :].reshape(-1),
            p_static[safe_sample_idx, :, :].reshape(-1),
            p_ours[safe_sample_idx, :, :].reshape(-1),
        ]
    )
    y_min = float(np.nanmin(y_pool))
    y_max = float(np.nanmax(y_pool))
    pad = 0.04 * (y_max - y_min + 1e-6)
    y_lim = (y_min - pad, y_max + pad)

    per_page = int(per_page) if int(per_page) > 0 else 16
    grid = int(np.ceil(np.sqrt(per_page)))
    page_count = int(np.ceil(n_nodes / per_page))
    if max_pages is not None:
        page_count = min(page_count, int(max_pages))

    for page in range(page_count):
        start = page * per_page
        end = min((page + 1) * per_page, n_nodes)
        nodes = list(range(start, end))

        fig, axes = plt.subplots(grid, grid, figsize=(14, 10), sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for ax_i, ax in enumerate(axes):
            if ax_i >= len(nodes):
                ax.axis("off")
                continue
            node_id = nodes[ax_i]
            ax.plot(x, t_real[safe_sample_idx, :, node_id], **style_true)
            ax.plot(x, p_gru[safe_sample_idx, :, node_id], **style_gru)
            ax.plot(x, p_static[safe_sample_idx, :, node_id], **style_static)
            ax.plot(x, p_ours[safe_sample_idx, :, node_id], **style_ours)
            ax.set_title(f"Node {node_id}", fontsize=9, pad=2)
            ax.set_ylim(*y_lim)
            ax.grid(True, alpha=0.15, linestyle="--")

        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["Raw", "GRU", "Static-GCN", "Ours"], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, fontsize=9)
        fig.suptitle(f"All nodes case curves (sample={safe_sample_idx})  Page {page+1}/{page_count}", y=1.05, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(str(out_dir / f"fig_all_nodes_cases_page_{page+1:03d}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_n_nodes_grid_single_png(
    pred_gru, pred_static, pred_ours, true_data, scaler_mean, scaler_std,
    save_path="outputs/fig_9_nodes_cases_3x3.png",
    sample_idx=50,
    node_ids=None,
    n_nodes_to_plot=9,
):
    for name, arr in [("pred_gru", pred_gru), ("pred_static", pred_static), ("pred_ours", pred_ours), ("true_data", true_data)]:
        if not hasattr(arr, "shape") or len(arr.shape) != 3:
            raise ValueError(f"{name} must have shape (N, H, V).")
    if pred_gru.shape != true_data.shape or pred_static.shape != true_data.shape or pred_ours.shape != true_data.shape:
        raise ValueError("Prediction/target shapes are inconsistent.")

    flow_mean = scaler_mean.flatten()[0] if hasattr(scaler_mean, "flatten") else scaler_mean
    flow_std = scaler_std.flatten()[0] if hasattr(scaler_std, "flatten") else scaler_std

    t_real = true_data * flow_std + flow_mean
    p_gru = pred_gru * flow_std + flow_mean
    p_static = pred_static * flow_std + flow_mean
    p_ours = pred_ours * flow_std + flow_mean

    n_samples, horizon, n_nodes = t_real.shape
    safe_sample_idx = int(np.clip(sample_idx, 0, n_samples - 1))
    n_nodes_to_plot = int(n_nodes_to_plot) if int(n_nodes_to_plot) > 0 else 9

    node_vol = np.std(t_real.reshape(-1, n_nodes), axis=0)
    order = np.argsort(node_vol)
    if node_ids is None:
        qs = np.linspace(0.05, 0.95, n_nodes_to_plot) if n_nodes_to_plot > 1 else np.array([0.5])
        chosen = [int(order[int(q * (n_nodes - 1))]) for q in qs]
    else:
        chosen = [int(np.clip(i, 0, n_nodes - 1)) for i in list(node_ids)[:n_nodes_to_plot]]
        while len(chosen) < n_nodes_to_plot:
            chosen.append(chosen[-1] if chosen else 0)

    unique = []
    for nid in chosen:
        if nid not in unique:
            unique.append(nid)
    for cand in order.tolist():
        if len(unique) >= n_nodes_to_plot:
            break
        if int(cand) not in unique:
            unique.append(int(cand))
    chosen = unique[:n_nodes_to_plot]

    y_pool = np.concatenate(
        [
            t_real[safe_sample_idx, :, chosen].reshape(-1),
            p_gru[safe_sample_idx, :, chosen].reshape(-1),
            p_static[safe_sample_idx, :, chosen].reshape(-1),
            p_ours[safe_sample_idx, :, chosen].reshape(-1),
        ]
    )
    y_min = float(np.nanmin(y_pool))
    y_max = float(np.nanmax(y_pool))
    pad = 0.05 * (y_max - y_min + 1e-6)
    y_lim = (y_min - pad, y_max + pad)

    c_true = "#1f77b4"
    c_gru = "#00A087"
    c_static = "#4DBBD5"
    c_ours = "#E64B35"
    style_true = dict(color=c_true, linewidth=1.6, linestyle="-")
    style_gru = dict(color=c_gru, linewidth=1.2, linestyle="--", alpha=0.95)
    style_static = dict(color=c_static, linewidth=1.2, linestyle=":", alpha=0.95)
    style_ours = dict(color=c_ours, linewidth=1.6, linestyle="-")

    x = np.arange(1, horizon + 1)
    grid = int(np.ceil(np.sqrt(n_nodes_to_plot)))
    fig, axes = plt.subplots(grid, grid, figsize=(14, 10), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(chosen):
            ax.axis("off")
            continue
        node_id = chosen[ax_i]
        ax.plot(x, t_real[safe_sample_idx, :, node_id], **style_true)
        ax.plot(x, p_gru[safe_sample_idx, :, node_id], **style_gru)
        ax.plot(x, p_static[safe_sample_idx, :, node_id], **style_static)
        ax.plot(x, p_ours[safe_sample_idx, :, node_id], **style_ours)
        ax.set_title(f"Node {node_id}", fontsize=10, pad=2)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.15, linestyle="--")

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Raw", "GRU", "Static-GCN", "Ours"], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, fontsize=10)
    fig.suptitle(f"{n_nodes_to_plot}-Node Case Curves (sample={safe_sample_idx})", y=1.06, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=600, bbox_inches="tight")
    plt.close(fig)

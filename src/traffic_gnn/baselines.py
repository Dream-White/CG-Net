import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_window = int(cfg.output_window)
        input_dim = cfg.traffic_dim + cfg.weather_dim
        self.gru = nn.GRU(input_dim, cfg.hidden_dim, batch_first=True)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.output_window)

    def forward(self, x, t_idx=None):
        bsz, t, n, num_f = x.shape
        x_reshaped = x.view(bsz * n, t, num_f)
        out, _ = self.gru(x_reshaped)
        out = self.fc(out[:, -1, :])
        return out.view(bsz, self.output_window, n).permute(0, 1, 2)


class StaticGCNBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_fc = nn.Linear(cfg.traffic_dim + cfg.weather_dim, cfg.hidden_dim)
        self.node_emb = nn.Parameter(torch.randn(cfg.num_nodes, 10))
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.gcn_fc = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.output_fc = nn.Linear(cfg.hidden_dim, cfg.output_window)

    def forward(self, x, t_idx=None):
        bsz, t, n, num_f = x.shape
        x_in = self.input_fc(x).view(bsz * n, t, -1)
        _, h_n = self.gru(x_in)
        last_hidden = h_n[-1].view(bsz, n, -1)
        adj = F.softmax(F.relu(torch.mm(self.node_emb, self.node_emb.t())), dim=-1)
        gcn_out = F.relu(self.gcn_fc(torch.matmul(adj, last_hidden))) + last_hidden
        out = self.output_fc(gcn_out)
        return out.permute(0, 2, 1)


def train_baseline(model, name: str, train_loader, test_loader, cfg, epochs: int = 30):
    print(f"Training baseline: {name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, y, t_idx in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, t_idx in test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            preds.append(model(x).cpu().numpy())
            trues.append(y.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

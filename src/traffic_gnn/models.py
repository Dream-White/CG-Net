import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAdaptiveGraph(nn.Module):
    def __init__(self, num_nodes: int, node_emb_dim: int, time_emb_dim: int):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb_dim))
        self.time_emb = nn.Embedding(288, time_emb_dim)
        self.gen_layer = nn.Linear(time_emb_dim, node_emb_dim)

    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(time_idx)
        t_effect = self.gen_layer(t_emb)
        dynamic_emb = self.node_emb.unsqueeze(0) + t_effect.unsqueeze(1)
        scores = torch.matmul(dynamic_emb, dynamic_emb.transpose(-1, -2))
        return F.softmax(F.relu(scores), dim=-1)


class CGUncertaintyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.traffic_fc = nn.Linear(cfg.traffic_dim, cfg.hidden_dim)
        self.weather_fc = nn.Linear(cfg.weather_dim, cfg.hidden_dim)
        self.fusion_gate = nn.Linear(cfg.hidden_dim * 2, 1)
        self.graph_gen = TemporalAdaptiveGraph(cfg.num_nodes, cfg.node_emb_dim, cfg.time_emb_dim)
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.gcn_fc = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(cfg.hidden_dim, cfg.output_window * 2)

    def forward(self, x: torch.Tensor, time_idx: torch.Tensor):
        bsz, t, n, _ = x.shape
        x_traffic = x[..., : self.cfg.traffic_dim]
        x_weather = x[..., 3:]

        h_t = F.relu(self.traffic_fc(x_traffic))
        h_w = F.relu(self.weather_fc(x_weather))
        z = torch.sigmoid(self.fusion_gate(torch.cat([h_t, h_w], dim=-1)))
        x_fused = z * h_t + (1.0 - z) * h_w

        x_in = x_fused.view(bsz * n, t, -1)
        _, h_n = self.gru(x_in)
        last_hidden = h_n[-1].view(bsz, n, -1)

        adj = self.graph_gen(time_idx)
        gcn_out = torch.matmul(adj, last_hidden)
        gcn_out = self.dropout(F.relu(self.gcn_fc(gcn_out))) + last_hidden

        out = self.output_fc(gcn_out)
        mu, log_var = torch.split(out, self.cfg.output_window, dim=-1)
        return mu.permute(0, 2, 1), log_var.permute(0, 2, 1)

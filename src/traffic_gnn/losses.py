import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridPhysicsLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 0.01, gamma: float = 0.01, weather_mod: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.weather_mod = bool(weather_mod)

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, x_context: torch.Tensor = None):
        var = torch.exp(torch.clamp(log_var, max=10))
        loss_nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var).mean()

        fft_pred = torch.fft.rfft(mu, dim=1)
        fft_target = torch.fft.rfft(target, dim=1)
        loss_fft = torch.mean(torch.abs(fft_pred - fft_target))
        loss = loss_nll + self.alpha * loss_fft

        vmin, vmax = -3.0, 3.0
        loss_cap = (F.relu(mu - vmax) + F.relu(vmin - mu)).mean()
        loss = loss + self.beta * loss_cap

        if mu.shape[1] > 1:
            dt = torch.abs(mu[:, 1:, :] - mu[:, :-1, :])
            if self.weather_mod and x_context is not None and x_context.dim() == 4 and x_context.shape[-1] >= 7:
                prec = x_context[..., 5].mean(dim=(1, 2), keepdim=True)
                vis = x_context[..., 6].mean(dim=(1, 2), keepdim=True)
                weight = (1.0 + 0.5 * torch.sigmoid(prec) + 0.5 * torch.sigmoid(F.relu(-vis))).view(-1, 1, 1)
                dt = dt * weight
            loss = loss + self.gamma * dt.mean()

        return loss

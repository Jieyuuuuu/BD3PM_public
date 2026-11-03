import torch
import torch.nn as nn

PRIOR = (0.014, 0.026)


def alpha_bar_schedule(timesteps, device="cpu"):
    return torch.linspace(1.0, 0.001, timesteps, device=device)


def per_channel(values, device="cpu", num_channels=2, pitch_dim=88):
    out = torch.empty(1, 1, num_channels * pitch_dim, device=device)
    out[..., 0::2] = values[0]
    out[..., 1::2] = values[1]
    return out


class BinaryDiffusion(nn.Module):
    def __init__(self, model, prior=PRIOR, timesteps=512):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.register_buffer("alpha_bar", alpha_bar_schedule(timesteps))
        self.register_buffer("prior", per_channel(prior))

    def forward(self, x0):
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        xt = torch.bernoulli(torch.clamp(x0 * ab + self.prior * (1 - ab), 0.0, 1.0))
        return self.model(xt, t)


@torch.no_grad()
def sample(model, shape, temperature=(0.3, 0.5), prior=PRIOR, timesteps=512, tail=50, device="cuda"):
    ab = alpha_bar_schedule(timesteps, device)
    d = per_channel(prior, device)
    tau = per_channel(temperature, device)
    img = torch.bernoulli(torch.clamp(d.expand(*shape), 0.0, 1.0))
    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        p0 = torch.sigmoid(model(img, t) / tau)
        ab_prev = ab[i - 1] if i > 0 else torch.tensor(1.0, device=device)
        ratio = ab[i] / ab_prev
        p_theta = p0 * ab_prev + d * (1 - ab_prev)
        l11 = ratio + (1 - ratio) * d
        l10 = (1 - ratio) * d
        lik1 = torch.where(img > 0.5, l11, 1 - l11)
        lik0 = torch.where(img > 0.5, l10, 1 - l10)
        post1 = p_theta * lik1
        prob = post1 / (post1 + (1 - p_theta) * lik0 + 1e-8)
        if i > tail:
            img = torch.bernoulli(torch.clamp(torch.nan_to_num(prob, nan=0.5), 0.0, 1.0))
        else:
            img = prob
    return (img >= 0.5).float()

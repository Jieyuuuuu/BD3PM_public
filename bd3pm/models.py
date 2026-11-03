import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LocalWindowAttention(nn.Module):
    def __init__(self, d_model, window=8, nhead=4, dropout=0.1):
        super().__init__()
        self.window = window
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        w = self.window
        pad = (w - L % w) % w
        x_pad = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
        L_pad = x_pad.shape[1]
        nw = L_pad // w
        x_win = x_pad.view(B, nw, w, D).reshape(B * nw, w, D)
        q, k, v = self.qkv(x_win).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B * nw, w, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nw, w, D).view(B, L_pad, D)
        if pad > 0:
            out = out[:, :L, :]
        return self.norm(x + self.dropout(self.proj(out)))


class PitchHarmonyConv(nn.Module):
    def __init__(self, pitch_dim=88, num_channels=2, kernel=5):
        super().__init__()
        self.pitch_dim = pitch_dim
        self.num_channels = num_channels
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=kernel,
                              padding=kernel // 2, groups=num_channels)
        self.norm = nn.LayerNorm(num_channels * pitch_dim)

    def forward(self, x):
        B, L, C = x.shape
        out = self.conv(x.view(B * L, self.num_channels, self.pitch_dim)).view(B, L, C)
        return self.norm(x + out)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, pitch_dim=88, num_channels=2,
                 window=8, dropout=0.1):
        super().__init__()
        flat = num_channels * pitch_dim
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(d_model, d_model * 2)
        self.ln_fwd = nn.LayerNorm(d_model)
        self.ln_bwd = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.refine_proj_down = nn.Linear(d_model, flat)
        self.pitch_conv = PitchHarmonyConv(pitch_dim, num_channels, kernel=5)
        self.win_attn = LocalWindowAttention(flat, window=window, nhead=4, dropout=dropout)
        self.refine_proj_up = nn.Linear(flat, d_model)
        self.refine_norm = nn.LayerNorm(d_model)

    def forward(self, x, t_emb):
        scale, shift = self.time_proj(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        out_fwd = self.mamba_fwd(h)
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(h, [1])), [1])
        out = 0.5 * (self.ln_fwd(x + self.dropout(out_fwd)) + self.ln_bwd(x + self.dropout(out_bwd)))
        r = self.refine_proj_up(self.win_attn(self.pitch_conv(self.refine_proj_down(out))))
        return self.refine_norm(out + r)


class BD3PM(nn.Module):
    def __init__(self, d_model=512, n_layers=12, d_state=64, d_conv=4, expand=2, num_channels=2,
                 pitch_dim=88, window=8, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.pitch_dim = pitch_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.spatial_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels)
        self.input_proj = nn.Sequential(nn.Linear(num_channels * pitch_dim, d_model), nn.SiLU())
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, pitch_dim, num_channels, window, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head_onset = nn.Linear(d_model, pitch_dim)
        self.output_head_sustain = nn.Linear(d_model, pitch_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        B, L, _ = x.shape
        x_2d = x.view(B, L, self.pitch_dim, self.num_channels).permute(0, 3, 1, 2).contiguous()
        x_2d = self.spatial_conv(x_2d) + x_2d
        h = self.input_proj(x_2d.permute(0, 2, 3, 1).reshape(B, L, -1))
        for layer in self.layers:
            h = layer(h, t_emb)
        h = self.final_norm(h)
        return torch.stack([self.output_head_onset(h), self.output_head_sustain(h)], dim=-1).reshape(B, L, -1)

"""Differentiable Mamba-1 recurrence with explicit incoming and outgoing states."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _scan(decay, update):
    """Inclusive affine prefix scan with linear total work."""
    length = decay.shape[1]
    if length == 1:
        return decay, update
    paired = length // 2 * 2
    right = decay[:, 1:paired:2]
    odd_decay, odd_update = _scan(
        right * decay[:, :paired:2], update[:, 1:paired:2] + right * update[:, :paired:2])
    left = decay[:, 2::2]
    count = left.shape[1]
    even_decay = torch.cat((decay[:, :1], left * odd_decay[:, :count]), dim=1)
    even_update = torch.cat((update[:, :1], update[:, 2::2] + left * odd_update[:, :count]), dim=1)
    def interleave(even, odd):
        n = odd.shape[1]
        joined = torch.stack((even[:, :n], odd), dim=2).flatten(1, 2)
        return torch.cat((joined, even[:, n:]), dim=1)
    return interleave(even_decay, odd_decay), interleave(even_update, odd_update)


class StatefulMamba(nn.Module):
    """Mamba-1 projections, depthwise convolution and selective SSM in PyTorch."""

    def __init__(self, config):
        super().__init__()
        width = config.channels
        self.inner = width * config.mamba_expand
        self.state_size = config.mamba_d_state
        self.conv_size = config.mamba_d_conv
        self.rank = math.ceil(width / 16)
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, self.conv_size, groups=self.inner)
        self.x_proj = nn.Linear(self.inner, self.rank + 2 * self.state_size, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        limit = self.rank ** -.5
        nn.init.uniform_(self.dt_proj.weight, -limit, limit)
        dt = torch.empty(self.inner).uniform_(math.log(.001), math.log(.1)).exp()
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, self.state_size + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.A_log._no_weight_decay = self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.inner, width, bias=False)
        self.initializer = nn.Sequential(nn.Linear(config.property_dim + 15, 32), nn.SiLU(),
                                         nn.Linear(32, self.inner * self.state_size))
        nn.init.zeros_(self.initializer[-1].weight)
        nn.init.zeros_(self.initializer[-1].bias)

    def allocate(self, environments):
        p = self.in_proj.weight
        return (p.new_zeros(environments, self.inner, self.conv_size),
                p.new_zeros(environments, self.inner, self.state_size, dtype=torch.float32))

    def initial_state(self, properties, reset_motion):
        convolution, _ = self.allocate(len(properties))
        ssm = self.initializer(torch.cat((properties, reset_motion), dim=-1))
        return convolution, ssm.reshape(-1, self.inner, self.state_size).float()

    def _tile(self, x, dt, b, c, incoming):
        dt = dt.float()
        decay = torch.exp(dt[..., None] * -self.A_log.float().exp())
        injected = (dt * x.float())[..., None] * b.float()[:, :, None]
        decay, injected = _scan(decay, injected)
        states = injected + decay * incoming[:, None]
        return (states * c.float()[:, :, None]).sum(-1), states[:, -1].contiguous()

    def forward(self, features, state=None, lengths=None):
        conv, ssm = self.allocate(len(features)) if state is None else state
        inputs, gate = self.in_proj(self.norm(features)).chunk(2, dim=-1)
        joined = torch.cat((conv[:, :, 1:].to(inputs.dtype), inputs.transpose(1, 2)), dim=-1)
        x = F.silu(self.conv(joined).transpose(1, 2))
        dt, b, c = self.x_proj(x).split((self.rank, self.state_size, self.state_size), dim=-1)
        dt = F.softplus(self.dt_proj(dt).float())
        if lengths is not None:
            active = torch.arange(x.shape[1], device=x.device) < lengths[:, None]
            dt = dt * active[..., None]
        outputs = []
        for start in range(0, x.shape[1], 32):
            sl = slice(start, start + 32)
            args = (x[:, sl], dt[:, sl], b[:, sl], c[:, sl], ssm)
            y, ssm = checkpoint(self._tile, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else self._tile(*args)
            outputs.append(y)
        y = (torch.cat(outputs, dim=1) + x.float() * self.D.float()) * F.silu(gate.float())
        conv = torch.cat((conv, inputs.transpose(1, 2)), dim=-1)
        if lengths is None:
            conv = conv[:, :, -self.conv_size:].contiguous()
        else:
            positions = lengths[:, None] + torch.arange(self.conv_size, device=x.device)
            conv = conv.gather(2, positions[:, None].expand(-1, self.inner, -1))
        return features + self.out_proj(y.to(x.dtype)), (conv, ssm)

    def step(self, features, states, valid):
        output, updated = self(features[:, None], states)
        for state, value in zip(states, updated):
            state.copy_(torch.where(valid[:, None, None], value, state))
        return features + getattr(self, "residual_scale", 1.) * (output[:, 0] - features)

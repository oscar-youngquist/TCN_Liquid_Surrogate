"""Recurrent residual layers for the FiLM pyramid."""
from types import SimpleNamespace

import torch
from torch import nn


class MambaMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError('Mamba memory requires the optional mamba-ssm CUDA dependency; '
                              'see docs/mamba_memory.md for installation.') from error
        self.norm = nn.LayerNorm(config.channels)
        self.block = Mamba(d_model=config.channels, d_state=config.mamba_d_state,
                           d_conv=config.mamba_d_conv, expand=config.mamba_expand,
                           use_fast_path=False, layer_idx=0)

    def forward(self, features, state=None):
        # The official differentiable scan starts at zero for each independent
        # clip. Burn-in remains in autograd; only its losses are discarded.
        normalized = self.norm(features)
        if state is None:
            return features + self.block(normalized)
        # First inference chunk uses the official scan and records its final
        # state; later chunks decode each NEW feature once with the official step.
        if state.seqlen_offset == 0:
            output = self.block(normalized, inference_params=state)
        else:
            states = state.key_value_memory_dict[0]
            output = torch.cat([self.block.step(normalized[:,i:i+1], *states)[0]
                                for i in range(normalized.shape[1])], dim=1)
        state.seqlen_offset += features.shape[1]
        return features + output

    def inference_state(self):
        return SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})

    def allocate(self, environments):
        return self.block.allocate_inference_cache(environments, max_seqlen=1)

    def step(self, features, states, valid):
        result, _, _ = self.block.step(self.norm(features)[:, None], *states)
        # All rows run together. Warmup rows have no persistent recurrent state;
        # their first contextual feature therefore starts from exactly zero.
        for state in states:
            state.masked_fill_(~valid[:, None, None], 0)
        return features + result[:, 0]


class MinGRUMemory(nn.Module):
    """Single-gate GRU with input-only projections and a parallel affine scan."""

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.channels)
        self.projection = nn.Linear(config.channels, 2 * config.channels)

    def _coefficients(self, features):
        gate, candidate = self.projection(self.norm(features)).chunk(2, dim=-1)
        if gate.dtype in (torch.bfloat16, torch.float16):
            gate, candidate = gate.float(), candidate.float()
        # Evaluate both gate weights directly to retain precision near saturation.
        return (-gate).sigmoid(), gate.sigmoid() * candidate

    def forward(self, features, state=None):
        carry, hidden = self._coefficients(features)
        # Compose h -> carry * h + hidden over prefixes of doubling length.
        offset = 1
        while offset < features.shape[1]:
            hidden = torch.cat((hidden[:, :offset],
                                hidden[:, offset:] + carry[:, offset:] * hidden[:, :-offset]), dim=1)
            carry = torch.cat((carry[:, :offset],
                               carry[:, offset:] * carry[:, :-offset]), dim=1)
            offset *= 2
        if state is not None:
            if state.hidden is not None:
                hidden = hidden + carry * state.hidden[:, None]
            state.hidden = hidden[:, -1].clone()
        return features + hidden.to(features.dtype)

    def inference_state(self):
        return SimpleNamespace(hidden=None)

    def allocate(self, environments):
        weight = self.projection.weight
        dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        return (weight.new_zeros(environments, self.norm.normalized_shape[0], dtype=dtype),)

    def step(self, features, states, valid):
        (state,) = states
        carry, hidden = self._coefficients(features)
        hidden = hidden + carry * state
        state.copy_(hidden.masked_fill(~valid[:, None], 0))
        return features + hidden.to(features.dtype)


def optimizer_parameters(model, weight_decay):
    """Honor official A_log/D no-decay flags, preserving legacy group layout."""
    excluded = [p for p in model.parameters() if getattr(p, '_no_weight_decay', False)]
    if not excluded:
        return model.parameters()
    return [dict(params=[p for p in model.parameters() if not getattr(p, '_no_weight_decay', False)],
                 weight_decay=weight_decay), dict(params=excluded, weight_decay=0.)]

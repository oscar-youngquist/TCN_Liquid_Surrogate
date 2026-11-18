"""Optional official Mamba residual; no alternate SSM implementation or reinit."""
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
            import torch
            states = state.key_value_memory_dict[0]
            output = torch.cat([self.block.step(normalized[:,i:i+1], *states)[0]
                                for i in range(normalized.shape[1])], dim=1)
        state.seqlen_offset += features.shape[1]
        return features + output

    def inference_state(self):
        from types import SimpleNamespace
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


def optimizer_parameters(model, weight_decay):
    """Honor official A_log/D no-decay flags, preserving legacy group layout."""
    excluded = [p for p in model.parameters() if getattr(p, '_no_weight_decay', False)]
    if not excluded:
        return model.parameters()
    return [dict(params=[p for p in model.parameters() if not getattr(p, '_no_weight_decay', False)],
                 weight_decay=weight_decay), dict(params=excluded, weight_decay=0.)]

"""Causal TCN and per-convolution streaming caches. Time is dimension 1 externally."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

HEADS = ('force', 'moment', 'r_B', 'v_B', 'S_B')


@dataclass
class ModelConfig:
    property_dim: int = 24
    embedding_dim: int = 32
    channels: int = 64
    kernel_size: int = 5
    dilations: tuple = (1, 2, 4, 8, 16)
    long_intervals: int = 40

    def __post_init__(self):
        self.dilations = tuple(self.dilations)
        if min(self.property_dim, self.embedding_dim, self.channels, self.kernel_size, self.long_intervals) < 1:
            raise ValueError('Model dimensions must be positive')
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError('Positive dilations required')

    @property
    def receptive_field(self):
        # Each causal convolution adds (kernel_size-1)*dilation past samples;
        # two convolutions per block give RF=249 with the default configuration.
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    @property
    def input_length(self):
        # RF samples yield the first fully contextual endpoint; another K
        # samples yield K more endpoints for K consecutive physics intervals.
        return self.receptive_field + self.long_intervals


class CausalConv(nn.Conv1d):
    @property
    def context(self):
        return (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x):
        # Left-only padding preserves time indices and forbids future inputs.
        # Training discards outputs whose receptive fields still touch padding.
        return super().forward(F.pad(x, (self.context, 0)))

    def step(self, x, cache):
        # Cache this convolution's INPUT activations, not its output. Keeping
        # (k-1)*d past samples reproduces its batch convolution at the next time.
        joined = torch.cat((cache, x), -1)
        y = F.conv1d(joined, self.weight, self.bias, dilation=self.dilation)
        return y, joined[..., -self.context:] if self.context else joined[..., :0]


class Block(nn.Module):
    def __init__(self, c, k, d):
        super().__init__()
        self.a = CausalConv(c, c, k, dilation=d)
        self.b = CausalConv(c, c, k, dilation=d)

    def forward(self, x):
        return x + F.silu(self.b(F.silu(self.a(x))))


class LiquidTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.embedding = nn.Sequential(nn.Linear(c.property_dim, c.embedding_dim), nn.SiLU(),
                                       nn.Linear(c.embedding_dim, c.embedding_dim), nn.SiLU())
        self.projection = nn.Conv1d(15 + c.embedding_dim, c.channels, 1)
        self.blocks = nn.ModuleList([Block(c.channels, c.kernel_size, d) for d in c.dilations])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(c.channels, c.channels), nn.SiLU(),
                                                nn.Linear(c.channels, 3)) for _ in HEADS])

    def forward(self, motion, properties, contextual_only=True):
        # motion is [B,T,15]; static recorded properties are [B,24]. Broadcasting
        # the property embedding over time introduces no future motion information.
        # Conv1d uses [B,C,T], while the public interface uses time dimension 1.
        embedding = self.embedding(properties)[:, None].expand(-1, motion.shape[1], -1)
        x = self.projection(torch.cat((motion, embedding), -1).transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        if contextual_only:
            if x.shape[1] < self.config.receptive_field:
                raise ValueError('Incomplete causal history')
            x = x[:, self.config.receptive_field - 1:]
        # Independent heads share the temporal features: output is [B,N,5,3]
        # in normalized units (N=41 for the default contextual training pass).
        return torch.stack([head(x) for head in self.heads], -2)


class CachedInference:
    """Physical-unit 12-value interface: [force, moment, r_B, v_B].

    Caller supplies one consecutive 5 ms sample per environment, and must reset
    on gaps, invalid held inputs, or payload changes. Invalid warmup is NaN,
    never an apparently usable padded prediction. No state is shared across envs.
    """
    def __init__(self, model, normalization, environments):
        self.model = model.eval()
        self.norm = normalization
        p = next(model.parameters())
        self.age = torch.zeros(environments, dtype=torch.long, device=p.device)
        self.caches = [p.new_zeros(environments, model.config.channels, conv.context)
                       for block in model.blocks for conv in (block.a, block.b)]

    def reset(self, environments=None):
        ids = slice(None) if environments is None else environments
        self.age[ids] = 0
        for cache in self.caches:
            cache[ids] = 0

    @torch.no_grad()
    def step(self, motion, properties, reset=None):
        if reset is not None:
            self.reset(reset)
        if not torch.isfinite(motion).all() or not torch.isfinite(properties).all():
            raise FloatingPointError('Nonfinite inference inputs')
        motion = self.norm.normalize('motion', motion)
        properties = self.norm.normalize('properties', properties)
        x = self.model.projection(torch.cat((motion, self.model.embedding(properties)), -1)[..., None])
        for i, block in enumerate(self.model.blocks):
            a, self.caches[2*i] = block.a.step(x, self.caches[2*i])
            b, self.caches[2*i+1] = block.b.step(F.silu(a), self.caches[2*i+1])
            x = x + F.silu(b)
        out = torch.stack([head(x[..., 0]) for head in self.model.heads], -2)
        out = self.norm.denormalize('targets', out)
        self.age += 1
        # Zeros initialize caches after a reset but never constitute valid
        # history: withhold predictions until this environment has RF real samples.
        valid = self.age >= self.model.config.receptive_field
        out = out[:, :4].flatten(1)
        return torch.where(valid[:, None], out, torch.full_like(out, float('nan'))), valid

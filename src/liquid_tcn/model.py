"""Causal TCN and per-convolution streaming caches. Time is dimension 1 externally."""
from dataclasses import dataclass, asdict
import math
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
    architecture: str = 'tcn'
    property_conditioning: str = 'existing'
    sample_seconds: float = .005
    attention_ages_ms: tuple | None = None
    attention_dim: int | None = None
    attention_heads: int | None = None
    attention_backend: str = 'custom'
    attention_dropout: float = 0.
    attention_projection_bias: bool = False
    attention_alpha: float = .1

    def __post_init__(self):
        self.dilations = tuple(self.dilations)
        if self.property_conditioning not in ('existing', 'film'):
            raise ValueError('property_conditioning must be existing or film')
        if self.architecture not in ('tcn', 'tcn_pyramid_attention'):
            raise ValueError('Unknown model architecture: ' + self.architecture)
        if not math.isfinite(self.sample_seconds) or self.sample_seconds <= 0:
            raise ValueError('sample_seconds must be finite and positive')
        if self.architecture == 'tcn_pyramid_attention':
            if self.attention_ages_ms is None:
                if len(self.dilations) != 5:
                    raise ValueError('Supply attention_ages_ms for a backbone with other than five blocks')
                self.attention_ages_ms = ((0,100,200,300,400,500,600,800,1000,1200),
                    (0,100,200,300,400,600,800,1000), (0,200,400,600,800), (0,200,400,600), (0,))
            self.attention_ages_ms = tuple(tuple(ages) for ages in self.attention_ages_ms)
            if len(self.attention_ages_ms) != len(self.dilations) or not any(self.attention_ages_ms):
                raise ValueError('Provide one age list per residual block and at least one token')
            for ages in self.attention_ages_ms:
                for age in ages:
                    steps = age / (1000 * self.sample_seconds)
                    if not math.isfinite(steps) or steps < 0 or not math.isclose(steps, round(steps), abs_tol=1e-7):
                        raise ValueError('Attention ages must be nonnegative integer multiples of sample_seconds')
            self.attention_dim = self.channels if self.attention_dim is None else self.attention_dim
            self.attention_heads = (4 if self.attention_dim % 4 == 0 else 1) if self.attention_heads is None else self.attention_heads
            if self.attention_dim < 1 or self.attention_heads < 1 or self.attention_dim % self.attention_heads:
                raise ValueError('Attention dimension must be divisible by the head count')
            if self.attention_backend not in ('custom', 'sdpa') or not 0 <= self.attention_dropout < 1:
                raise ValueError('Invalid attention backend or dropout')
            if not math.isfinite(self.attention_alpha):
                raise ValueError('attention_alpha must be finite')
        if min(self.property_dim, self.embedding_dim, self.channels, self.kernel_size, self.long_intervals) < 1:
            raise ValueError('Model dimensions must be positive')
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError('Positive dilations required')

    @property
    def block_receptive_fields(self):
        # Each block has two causal convolutions, both at the block dilation.
        total = 1
        fields = []
        for dilation in self.dilations:
            total += 2 * (self.kernel_size - 1) * dilation
            fields.append(total)
        return tuple(fields)

    @property
    def attention_lags(self):
        return tuple(tuple(round(age / (1000 * self.sample_seconds)) for age in ages)
                     for ages in self.attention_ages_ms)

    @property
    def receptive_field(self):
        backbone = self.block_receptive_fields[-1]
        if self.architecture == 'tcn':
            return backbone
        # Include the deepest query even if its block has no memory tokens.
        return max(backbone, max(lag + rf for rf, lags in
                   zip(self.block_receptive_fields, self.attention_lags) for lag in lags))

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


class FiLM(nn.Module):
    """Zero-initialized final projection of the shared static property encoding.

    Direct zero Parameters avoid consuming initialization RNG, so the common
    backbone/attention weights match an existing-conditioning model at a fixed seed.
    """
    def __init__(self, embedding_dim, blocks, channels):
        super().__init__()
        self.blocks, self.channels = blocks, channels
        self.weight = nn.Parameter(torch.zeros(blocks * 2 * channels, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(blocks * 2 * channels))

    def forward(self, embedding):
        return F.linear(embedding, self.weight, self.bias).reshape(-1, self.blocks, 2, self.channels)


class LiquidTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if type(self) is LiquidTCN and c.architecture != 'tcn':
            raise ValueError('Use build_model(config) to construct the selected architecture')
        self.embedding = nn.Sequential(nn.Linear(c.property_dim, c.embedding_dim), nn.SiLU(),
                                       nn.Linear(c.embedding_dim, c.embedding_dim), nn.SiLU())
        self.projection = nn.Conv1d(15 + c.embedding_dim, c.channels, 1)
        self.blocks = nn.ModuleList([Block(c.channels, c.kernel_size, d) for d in c.dilations])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(c.channels, c.channels), nn.SiLU(),
                                                  nn.Linear(c.channels, c.channels), nn.SiLU(),
                                                  nn.Linear(c.channels, 3)) for _ in HEADS])
        self.film = (FiLM(c.embedding_dim, len(c.dilations), c.channels)
                     if c.property_conditioning == 'film' else None)

    def encode_properties(self, properties):
        embedding = self.embedding(properties)
        return embedding, self.film(embedding) if self.film is not None else None

    @staticmethod
    def condition_block(x, modulation, index):
        if modulation is None:
            return x
        gamma, beta = modulation[:, index].unbind(1)
        return (1 + gamma[..., None]) * x + beta[..., None]

    def forward(self, motion, properties, contextual_only=True):
        # motion is [B,T,15]; static recorded properties are [B,24]. Broadcasting
        # the property embedding over time introduces no future motion information.
        # Conv1d uses [B,C,T], while the public interface uses time dimension 1.
        embedding, modulation = self.encode_properties(properties)
        embedding = embedding[:, None].expand(motion.shape[0], motion.shape[1], -1)
        x = self.projection(torch.cat((motion, embedding), -1).transpose(1, 2))
        for i, block in enumerate(self.blocks):
            x = self.condition_block(block(x), modulation, i)
        x = x.transpose(1, 2)
        if contextual_only:
            if x.shape[1] < self.config.receptive_field:
                raise ValueError('Incomplete causal history')
            x = x[:, self.config.receptive_field - 1:]
        # Independent heads share the temporal features: output is [B,N,5,3]
        # in normalized units (N=41 for the default contextual training pass).
        return torch.stack([head(x) for head in self.heads], -2)


class ReferenceCachedInference:
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
        self.cursor = 0
        self.histories = ([p.new_zeros(environments, max(lags)+1, model.config.channels)
                           if lags else None for lags in model.config.attention_lags]
                          if model.config.architecture == 'tcn_pyramid_attention' else [])
        self.caches = [p.new_zeros(environments, model.config.channels, conv.context)
                       for block in model.blocks for conv in (block.a, block.b)]

    def reset(self, environments=None):
        ids = slice(None) if environments is None else environments
        self.age[ids] = 0
        for cache in self.caches + self.histories:
            if cache is not None:
                cache[ids] = 0

    @torch.no_grad()
    def step(self, motion, properties, reset=None, return_all=False):
        if reset is not None:
            self.reset(reset)
        finite = torch.isfinite(motion).all() & torch.isfinite(properties).all()
        if motion.is_cuda:
            torch._assert_async(finite, 'Nonfinite inference inputs')
        elif not finite:
            raise FloatingPointError('Nonfinite inference inputs')
        motion = self.norm.normalize('motion', motion)
        properties = self.norm.normalize('properties', properties)
        embedding, modulation = self.model.encode_properties(properties)
        embedding = embedding.expand(motion.shape[0], -1)
        x = self.model.projection(torch.cat((motion, embedding), -1)[..., None])
        memory = []
        for i, block in enumerate(self.model.blocks):
            a, self.caches[2*i] = block.a.step(x, self.caches[2*i])
            b, self.caches[2*i+1] = block.b.step(F.silu(a), self.caches[2*i+1])
            x = self.model.condition_block(x + F.silu(b), modulation, i)
            if self.histories and self.histories[i] is not None:
                ring = self.histories[i]
                ring[:, self.cursor % ring.shape[1]] = x[..., 0]
                lags = getattr(self.model, f'lags_{i}')
                memory.append(ring[:, (self.cursor - lags) % ring.shape[1]])
        if self.histories:
            fused = self.model.fuse(x[..., 0][:, None], torch.cat(memory, 1)[:, None])
            x = fused.transpose(1, 2)
        self.cursor += 1
        out = torch.stack([head(x[..., 0]) for head in self.model.heads], -2)
        out = self.norm.denormalize('targets', out)
        self.age += 1
        # Zeros initialize caches after a reset but never constitute valid
        # history: withhold predictions until this environment has RF real samples.
        valid = self.age >= self.model.config.receptive_field
        out = out.masked_fill(~valid[:, None, None], float('nan'))
        return (out if return_all else out[:, :4].flatten(1)), valid


def build_model(config=None):
    """One construction path for training, checkpoint loading and export.

    Missing architecture fields in older checkpoints retain the original module
    names and parameter shapes; no state-dict migration or partial loading occurs.
    """
    config = ModelConfig(**config) if isinstance(config, dict) else (config or ModelConfig())
    if config.architecture == 'tcn_pyramid_attention':
        from .pyramid import PyramidAttentionTCN
        return PyramidAttentionTCN(config)
    return LiquidTCN(config)


from .streaming import CachedInference  # Preserve the public inference import path.

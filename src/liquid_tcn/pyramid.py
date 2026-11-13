"""Sparse causal cross-attention over residual-block features.

Attention adapted from Sanjaya Subedi's scaled-dot-product and efficient
multi-head attention implementation:
https://sanjayasubedi.com.np/deeplearning/multihead-attention-from-scratch/
The K/V projection is fused, query dimensions are retained, and the output
feature dropout remains BEFORE the final projection (not on attention weights).
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import LiquidTCN


class SparseCrossAttention(nn.Module):
    def __init__(self, width, dimension, heads, dropout=0., bias=False, backend='custom'):
        super().__init__()
        self.heads, self.head_dim = heads, dimension // heads
        self.scale = self.head_dim ** -.5
        self.backend = backend
        self.q = nn.Linear(width, dimension, bias=bias)
        self.kv = nn.Linear(width, 2 * dimension, bias=bias)
        self.projection = nn.Linear(dimension, width, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, return_attention=False):
        # Leading dimensions may include both batch and prediction time. Each
        # time has its own Q queries and M historical tokens, never a T x T map.
        def split(x):
            # Flatten independent prediction times into the batch dimension;
            # four-dimensional tensors also permit fused SDPA CUDA kernels.
            return x.reshape(-1, x.shape[-2], self.heads, self.head_dim).transpose(1, 2)
        q = split(self.q(query))
        k, v = (split(x) for x in self.kv(memory).chunk(2, dim=-1))
        weights = None
        if self.backend == 'custom' or return_attention:
            weights = ((q @ k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
            attended = weights @ v
        else:
            # SDPA's weight dropout is intentionally zero: the supplied module
            # drops the combined output features, with the same mask semantics
            # in either backend. All memory is already causal; no triangle mask.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.,
                                                      is_causal=False, scale=self.scale)
        combined = attended.transpose(1, 2).flatten(-2).reshape(*query.shape[:-1], -1)
        output = self.projection(self.dropout(combined))
        if return_attention:
            weights = weights.reshape(*query.shape[:-2], self.heads, query.shape[-2], memory.shape[-2])
            return output, weights
        return output


class PyramidAttentionTCN(LiquidTCN):
    def __init__(self, config):
        super().__init__(config)
        c = config
        self.query_norm = nn.LayerNorm(c.channels)
        self.memory_norm = nn.LayerNorm(c.channels)
        self.block_embedding = nn.Embedding(len(c.dilations), c.channels)
        ages = sorted({lag for lags in c.attention_lags for lag in lags})
        self.age_embedding = nn.Embedding(len(ages), c.channels)
        blocks, age_ids = [], []
        for i, lags in enumerate(c.attention_lags):
            self.register_buffer(f'lags_{i}', torch.tensor(lags, dtype=torch.long), persistent=False)
            blocks.extend([i] * len(lags))
            age_ids.extend(ages.index(lag) for lag in lags)
        self.register_buffer('token_blocks', torch.tensor(blocks), persistent=False)
        self.register_buffer('token_ages', torch.tensor(age_ids), persistent=False)
        self.attention = SparseCrossAttention(c.channels, c.attention_dim, c.attention_heads,
            c.attention_dropout, c.attention_projection_bias, c.attention_backend)
        self.alpha = nn.Parameter(torch.tensor(c.attention_alpha))

    def fuse(self, query, memory, return_attention=False):
        # Tokens retain both scale identity and physical age. LayerNorm acts on
        # features only, never across times/environments, so remains causal.
        memory = self.memory_norm(memory) + self.block_embedding(self.token_blocks) + self.age_embedding(self.token_ages)
        result = self.attention(self.query_norm(query).unsqueeze(-2), memory, return_attention)
        attended, weights = result if return_attention else (result, None)
        fused = query + self.alpha * attended.squeeze(-2)
        return (fused, weights) if return_attention else fused

    def forward(self, motion, properties, contextual_only=True, return_attention=False):
        start = self.config.receptive_field - 1 if contextual_only else 0
        if motion.shape[1] <= start:
            raise ValueError('Incomplete causal history')
        times = torch.arange(start, motion.shape[1], device=motion.device)
        embedding, modulation = self.encode_properties(properties)
        embedding = embedding[:, None].expand(motion.shape[0], motion.shape[1], -1)
        x = self.projection(torch.cat((motion, embedding), -1).transpose(1, 2))
        memories = []
        for i, block in enumerate(self.blocks):
            x = self.condition_block(block(x), modulation, i)
            lags = getattr(self, f'lags_{i}')
            if lags.numel():
                # Gather first; project only selected features. The block loop is
                # independent of sequence length, and every query time is batched.
                positions = times[:, None] - lags[None]
                memories.append(x.transpose(1, 2)[:, positions.clamp_min(0)])
        result = self.fuse(x.transpose(1, 2)[:, start:], torch.cat(memories, dim=2), return_attention)
        features, weights = result if return_attention else (result, None)
        out = torch.stack([head(features) for head in self.heads], -2)
        if not contextual_only:
            # Diagnostic full-length output explicitly marks startup as unusable.
            out = out.masked_fill((times < self.config.receptive_field-1)[None, :, None, None], float('nan'))
        return (out, weights) if return_attention else out

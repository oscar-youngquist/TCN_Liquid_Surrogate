"""Current-input FiLM encoder and recurrent Mamba prediction heads."""
import torch
from torch import nn
from .model import LiquidTCN, FiLM
from .stateful_mamba import StatefulMamba


class RecurrentOnly(LiquidTCN):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList()
        self.film = FiLM(config.embedding_dim, 1, config.channels)
        self.temporal = StatefulMamba(config)

    def forward(self, motion, properties, memory_state=None, return_state=False, memory_lengths=None):
        embedding, modulation = self.encode_properties(properties)
        x = self.projection(torch.cat((motion, embedding[:, None].expand(-1, motion.shape[1], -1)), -1).transpose(1, 2))
        x = self.condition_block(x, modulation, 0).transpose(1, 2)
        x, state = self.temporal(x, memory_state, memory_lengths)
        out = torch.stack([head(x) for head in self.heads], -2)
        return (out, state) if return_state else out

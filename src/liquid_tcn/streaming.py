"""Bounded inference-only rings and reusable static payload conditioning.

Use set_properties once and step(motion) for the allocation/synchronization-light
path. The compatible step(motion, properties) form detects changed tensors too.
No detached state from this class participates in gradient-based training.
"""
import weakref
import torch
from torch.nn import functional as F


def version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # Inference tensors do not have mutation counters.
        return None


class CachedInference:
    def __init__(self, model, normalization, environments):
        if environments < 1:
            raise ValueError('environments must be positive')
        self.model = model.eval()
        self.norm = normalization
        self.environments = environments
        self._properties = None
        self._source = None
        self._stamp = None
        self.invalidate()

    def _model_signature(self):
        # Tensor versions are host metadata, not GPU reads. In-place optimizer
        # updates, load_state_dict, replacement tensors and .to() invalidate state.
        return (self.model.config.property_conditioning,
                tuple((id(p), version(p), p.device, p.dtype) for p in self.model.parameters()))

    @torch.inference_mode(False)
    def invalidate(self):
        """Rebuild after weight/device/dtype changes; next sample starts warmup.

        Also call explicitly after editing parameters created in inference_mode
        (which lack mutation counters) or replacing normalization statistics.
        """
        p = next(self.model.parameters())
        n, c = self.environments, self.model.config.channels
        self.age = torch.zeros(n, dtype=torch.long, device=p.device)
        self.cursor = 0
        self.inputs = p.new_zeros(n, 15+self.model.config.embedding_dim)
        self.caches, self.taps, self.tap_positions = [], [], []
        tap_buffers = {}
        for block in self.model.blocks:
            for conv in (block.a, block.b):
                length = conv.context+1
                self.caches.append(p.new_zeros(n, c, length))
                key = (c, conv.kernel_size[0])
                if key not in tap_buffers:
                    tap_buffers[key] = p.new_empty(n, *key)
                # Convolutions run sequentially on one stream, so all blocks can
                # reuse a single k-tap scratch buffer rather than retaining ten.
                self.taps.append(tap_buffers[key])
                # Chronological cross-correlation taps for every ring position.
                positions = torch.arange(length, device=p.device)[:, None]
                offsets = torch.arange(conv.kernel_size[0], device=p.device)*conv.dilation[0]-conv.context
                self.tap_positions.append((positions+offsets) % length)
        self.histories, self.memory_positions, self.memory_slices = [], [], []
        count = 0
        if self.model.config.architecture == 'tcn_pyramid_attention':
            for lags in self.model.config.attention_lags:
                length = max(lags)+1 if lags else 0
                self.histories.append(p.new_zeros(n, length, c) if lags else None)
                self.memory_positions.append((torch.arange(length, device=p.device)[:, None]
                    -torch.tensor(lags, device=p.device)) % length if lags else None)
                self.memory_slices.append(slice(count, count+len(lags)))
                count += len(lags)
        self.memory = p.new_empty(n, count, c) if count else None
        self.motion_mean, self.motion_std = self.norm._params('motion', p)
        self.target_mean, self.target_std = self.norm._params('targets', p)
        self.modulation = None
        self._film = None
        self._ready = False
        self._signature = self._model_signature()
        self._source = None
        self._stamp = None

    def _ensure_current(self):
        if self.model.training or self._model_signature() != self._signature:
            self.model.eval()
            self.invalidate()
        if not self._ready and self._properties is not None:
            self.set_properties(self._properties)

    @torch.inference_mode()
    def set_properties(self, properties, property_indices=None):
        """Register [P,D] payloads and optional [environments] payload indices.

        A single property row broadcasts without duplicating FiLM coefficients.
        Repeated rows are encoded once; bank/index mapping is resolved only here.
        Only environments whose physical properties changed lose their history.
        """
        self._ensure_model_only()
        p = next(self.model.parameters())
        source = properties
        properties = properties.to(device=p.device, dtype=p.dtype)
        if properties.ndim == 1:
            properties = properties[None]
        if properties.ndim != 2 or properties.shape[1] != self.model.config.property_dim:
            raise ValueError('Expected properties [payloads, property_dim]')
        if property_indices is not None:
            properties = properties[property_indices.to(device=p.device, dtype=torch.long)]
        if len(properties) not in (1, self.environments):
            raise ValueError('Provide one shared payload, one per environment, or property_indices')
        if not torch.isfinite(properties).all():
            raise FloatingPointError('Nonfinite inference properties')
        previous = self._properties
        if previous is not None:
            previous = previous.to(properties)
            changed = (properties.expand(self.environments, -1) != previous.expand(self.environments, -1)).any(-1)
            if self._ready and not bool(changed.any()):
                self._remember_source(source, property_indices)
                return
            self.reset(changed)
        self._properties = properties.clone()
        unique, inverse = torch.unique(properties, dim=0, return_inverse=True)
        encoded, modulation = self.model.encode_properties(self.norm.normalize('properties', unique))
        if len(unique) != 1:
            encoded = encoded[inverse]
            modulation = modulation[inverse] if modulation is not None else None
        self.inputs[:, 15:].copy_(encoded.expand(self.environments, -1))
        self.modulation = modulation
        self._film = (tuple(((1+modulation[:, i, 0])[..., None], modulation[:, i, 1, :, None])
                            for i in range(len(self.model.blocks))) if modulation is not None else None)
        self._ready = True
        self._remember_source(source, property_indices)

    def _remember_source(self, properties, indices):
        self._source = weakref.ref(properties) if indices is None else None
        self._stamp = version(properties)

    def _ensure_model_only(self):
        if self.model.training or self._model_signature() != self._signature:
            self.model.eval()
            self.invalidate()

    @torch.inference_mode()
    def reset(self, environments=None):
        if environments is None:
            self.age.zero_()
            for cache in self.caches+self.histories:
                if cache is not None:
                    cache.zero_()
            return
        mask = torch.zeros_like(self.age, dtype=torch.bool)
        mask[environments] = True
        self.age.masked_fill_(mask, 0)
        for cache in self.caches+self.histories:
            if cache is not None:
                cache.masked_fill_(mask[:, None, None], 0)

    def _conv(self, x, conv, index):
        ring = self.caches[index]
        pos = self.cursor % ring.shape[-1]
        ring[:, :, pos].copy_(x[..., 0])
        torch.index_select(ring, 2, self.tap_positions[index][pos], out=self.taps[index])
        # Gather only k taps, not a dilation-sized full window. Dilation is already
        # represented by the indices, so the small convolution uses dilation=1.
        return F.conv1d(self.taps[index], conv.weight, conv.bias)

    @torch.inference_mode()
    def step(self, motion, properties=None, reset=None, return_all=False):
        """Return physical [B,12] (or all five [B,5,3] heads) and validity.

        Existing callers may pass properties each step. Reuse a versioned tensor
        or call set_properties once and omit it to avoid value comparison/sync.
        Property mutations in ordinary tensors are detected by their version;
        inference tensors without versions require a value comparison.
        """
        self._ensure_current()
        if properties is not None:
            unchanged = (self._ready and self._source is not None and self._source() is properties
                         and self._stamp is not None and self._stamp == version(properties))
            if not unchanged:
                self.set_properties(properties)
        if not self._ready:
            raise ValueError('Call set_properties or supply properties to step')
        if reset is not None:
            self.reset(reset)
        if motion.shape != (self.environments, 15):
            raise ValueError('Expected one motion sample per environment: [environments,15]')
        finite = torch.isfinite(motion).all()
        if motion.is_cuda:
            torch._assert_async(finite, 'Nonfinite inference inputs')
        elif not finite:
            raise FloatingPointError('Nonfinite inference inputs')
        torch.sub(motion, self.motion_mean, out=self.inputs[:, :15])
        self.inputs[:, :15].div_(self.motion_std)
        x = self.model.projection(self.inputs[..., None])
        for i, block in enumerate(self.model.blocks):
            a = self._conv(x, block.a, 2*i)
            b = self._conv(F.silu(a), block.b, 2*i+1)
            x = x + F.silu(b)
            if self._film is not None:
                scale, shift = self._film[i]
                x = scale*x + shift
            if self.histories and self.histories[i] is not None:
                ring = self.histories[i]
                pos = self.cursor % ring.shape[1]
                ring[:, pos].copy_(x[..., 0])
                # out= reuses the selected-token buffer; no cat or roll. Features,
                # not projected K/V, are cached because age-dependent embeddings
                # must enter the projections anew at each query timestamp.
                torch.index_select(ring, 1, self.memory_positions[i][pos],
                                   out=self.memory[:, self.memory_slices[i]])
        if self.memory is not None:
            x = self.model.fuse(x[..., 0][:, None], self.memory[:, None]).transpose(1, 2)
        out = torch.stack([head(x[..., 0]) for head in self.model.heads], -2)
        out.mul_(self.target_std).add_(self.target_mean)
        self.cursor += 1
        self.age.add_(1)
        valid = self.age >= self.model.config.receptive_field
        out.masked_fill_(~valid[:, None, None], float('nan'))
        return (out if return_all else out[:, :4].flatten(1)), valid

    def persistent_bytes(self):
        """Unique storage bytes held by this cache, excluding model parameters."""
        seen = {}
        def visit(value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                seen[(value.device, storage.data_ptr())] = storage.nbytes()
            elif isinstance(value, (list, tuple)):
                for item in value: visit(item)
        for value in vars(self).values(): visit(value)
        return sum(seen.values())

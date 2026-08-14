"""Frozen M2-B-A A0 single-depth causal memory adapter.

This module is deliberately independent of the vendored Wan package.  The
caller supplies Wan block-0's *modulated self-attention input* and the frozen
pre-head hidden state.  Empty, rejected, and disabled routes are physical
bypasses: no adapter submodule is evaluated and the exact ``h_base`` object is
returned.

A0 only defines and instruments the frozen tensor contract.  It contains no
training loop and makes no A1 viability claim.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class CausalMemoryAdapterConfig:
    latent_channels: int = 16
    memory_frames: int = 3
    latent_height: int = 58
    latent_width: int = 104
    encoder_channels: int = 320
    pool_height: int = 4
    pool_width: int = 8
    hidden_dim: int = 5120
    num_heads: int = 40
    eps: float = 1e-6
    alpha: float = 0.05
    init_seed: int = 42

    @property
    def head_dim(self) -> int:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return self.hidden_dim // self.num_heads

    @property
    def memory_tokens(self) -> int:
        return self.memory_frames * self.pool_height * self.pool_width

    def canonical_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


class WanCompatibleRMSNorm(nn.Module):
    """Wan RMSNorm semantics without importing the CUDA-heavy Wan package."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normed.to(x.dtype) * self.weight


def _sinusoidal_partition(length: int, dim: int, device: torch.device) -> Tensor:
    """Standard fixed sin/cos encoding; an odd final coordinate stays zero."""
    result = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pairs = dim // 2
    if pairs == 0:
        return result
    positions = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(pairs, dtype=torch.float32, device=device)
        / max(pairs, 1)
    )[None, :]
    angles = positions * frequencies
    result[:, 0 : 2 * pairs : 2] = torch.sin(angles)
    result[:, 1 : 2 * pairs : 2] = torch.cos(angles)
    return result


def fixed_separable_position_encoding(config: CausalMemoryAdapterConfig, device: torch.device) -> Tensor:
    """Return frozen frame/row/column PE in frame-major,row-major,col-major order."""
    # The frozen 5120-D contract partitions as 1708/1706/1706.  The same
    # deterministic remainder allocation makes reduced-dimension CPU fixtures
    # possible without changing production semantics.
    base_dim, remainder = divmod(config.hidden_dim, 3)
    time_dim = base_dim + remainder
    row_dim = base_dim
    col_dim = base_dim
    time = _sinusoidal_partition(config.memory_frames, time_dim, device)
    row = _sinusoidal_partition(config.pool_height, row_dim, device)
    col = _sinusoidal_partition(config.pool_width, col_dim, device)
    values = []
    for frame_index in range(config.memory_frames):
        for row_index in range(config.pool_height):
            for col_index in range(config.pool_width):
                values.append(torch.cat((time[frame_index], row[row_index], col[col_index])))
    return torch.stack(values)


class OrderedLatentMemoryEncoder(nn.Module):
    def __init__(self, config: CausalMemoryAdapterConfig) -> None:
        super().__init__()
        self.config = config
        # Isolate initialization so construction does not perturb caller RNG.
        devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.init_seed)
            self.conv = nn.Conv2d(
                config.latent_channels,
                config.encoder_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )
            self.projection = nn.Linear(config.encoder_channels, config.hidden_dim, bias=True)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        self.pool = nn.AdaptiveAvgPool2d((config.pool_height, config.pool_width))
        self.norm = WanCompatibleRMSNorm(config.hidden_dim, config.eps)

    def forward(self, memory_latents: Tensor) -> Tensor:
        cfg = self.config
        expected = (
            cfg.latent_channels,
            cfg.memory_frames,
            cfg.latent_height,
            cfg.latent_width,
        )
        if memory_latents.ndim != 5 or tuple(memory_latents.shape[1:]) != expected:
            raise ValueError(
                "memory_latents must be ordered [B,C,F,H,W] with trailing shape "
                f"{expected}, got {tuple(memory_latents.shape)}"
            )
        batch = memory_latents.shape[0]
        # Frame-major ordering is made explicit before flattening spatial rows/cols.
        frames = memory_latents.permute(0, 2, 1, 3, 4).reshape(
            batch * cfg.memory_frames,
            cfg.latent_channels,
            cfg.latent_height,
            cfg.latent_width,
        )
        encoded = self.pool(F.silu(self.conv(frames)))
        encoded = encoded.permute(0, 2, 3, 1).reshape(
            batch, cfg.memory_tokens, cfg.encoder_channels
        )
        tokens = self.projection(encoded)
        pe = fixed_separable_position_encoding(cfg, tokens.device).to(tokens.dtype)
        return self.norm(tokens + pe.unsqueeze(0))


class DetachedRMSBridge(nn.Module):
    def __init__(self, config: CausalMemoryAdapterConfig) -> None:
        super().__init__()
        self.config = config
        bottleneck = config.hidden_dim // 4
        if bottleneck < 1:
            raise ValueError("hidden_dim must be at least 4")
        devices = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.init_seed)
            self.norm = WanCompatibleRMSNorm(config.hidden_dim, config.eps)
            self.w1 = nn.Linear(config.hidden_dim, bottleneck, bias=True)
            self.w2 = nn.Linear(bottleneck, config.hidden_dim, bias=True)
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.zeros_(self.w1.bias)
            nn.init.normal_(self.w2.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.w2.bias)

    def forward(self, memory_delta: Tensor, h_base: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raw = self.w2(F.silu(self.w1(self.norm(memory_delta))))
        base_rms = h_base.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        raw_rms = raw.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        scale = torch.minimum(
            torch.ones_like(raw_rms),
            self.config.alpha * base_rms / (raw_rms + self.config.eps),
        )
        capped = raw * scale.to(raw.dtype)
        return h_base + capped.to(h_base.dtype), {
            "raw_delta": raw,
            "capped_delta": capped,
            "cap_scale": scale,
        }


class CausalMemoryAdapter(nn.Module):
    """Route-specific block-0 memory read and pre-head residual bridge."""

    def __init__(
        self,
        config: CausalMemoryAdapterConfig,
        base_self_attention: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        for name in ("q", "k", "v", "o", "norm_q", "norm_k"):
            if not hasattr(base_self_attention, name):
                raise TypeError(f"base self-attention is missing {name}")

        # Frozen q/norm_q belong to the base model and intentionally must not be
        # registered in this adapter's state_dict.
        object.__setattr__(self, "_base_q", base_self_attention.q)
        object.__setattr__(self, "_base_norm_q", base_self_attention.norm_q)
        self.memory_encoder = OrderedLatentMemoryEncoder(config)
        self.wk_mem = copy.deepcopy(base_self_attention.k)
        self.wv_mem = copy.deepcopy(base_self_attention.v)
        self.wo_mem = copy.deepcopy(base_self_attention.o)
        self.norm_k_mem = copy.deepcopy(base_self_attention.norm_k)
        self.bridge = DetachedRMSBridge(config)

    @classmethod
    def from_wan_self_attention(
        cls,
        base_self_attention: nn.Module,
        config: Optional[CausalMemoryAdapterConfig] = None,
    ) -> "CausalMemoryAdapter":
        config = config or CausalMemoryAdapterConfig()
        if getattr(base_self_attention, "dim", config.hidden_dim) != config.hidden_dim:
            raise ValueError("base self-attention dim differs from adapter config")
        if getattr(base_self_attention, "num_heads", config.num_heads) != config.num_heads:
            raise ValueError("base self-attention head count differs from adapter config")
        return cls(config, base_self_attention)

    def _physical_bypass(
        self,
        h_base: Tensor,
        *,
        adapter_enabled: bool,
        memory_latents: Optional[Tensor],
        rejected: bool,
    ) -> bool:
        return (not adapter_enabled) or rejected or memory_latents is None or memory_latents.numel() == 0

    def needs_integration_hooks(
        self,
        *,
        adapter_enabled: bool,
        memory_latents: Optional[Tensor],
        rejected: bool,
    ) -> bool:
        """Production call-sites use this before installing Wan hooks."""
        return not self._physical_bypass(
            torch.empty(0), adapter_enabled=adapter_enabled,
            memory_latents=memory_latents, rejected=rejected,
        )

    def forward(
        self,
        h_sa0: Tensor,
        h_base: Tensor,
        *,
        memory_latents: Optional[Tensor],
        route_query_mask: Optional[Tensor],
        memory_validity_mask: Optional[Tensor] = None,
        adapter_enabled: bool = True,
        rejected: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor | bool]]:
        if self._physical_bypass(
            h_base,
            adapter_enabled=adapter_enabled,
            memory_latents=memory_latents,
            rejected=rejected,
        ):
            if return_diagnostics:
                return h_base, {"physical_bypass": True}
            return h_base

        if h_sa0.shape != h_base.shape or h_base.ndim != 3:
            raise ValueError("h_sa0 and h_base must share [B,L,D]")
        batch, length, dim = h_base.shape
        if dim != self.config.hidden_dim:
            raise ValueError(f"hidden dim must be {self.config.hidden_dim}, got {dim}")
        if route_query_mask is None or route_query_mask.dtype != torch.bool:
            raise TypeError("route_query_mask must be a boolean [B,L] tensor")
        if tuple(route_query_mask.shape) != (batch, length):
            raise ValueError("route_query_mask shape differs from hidden layout")
        counts = route_query_mask.sum(dim=1)
        if counts.numel() == 0 or int(counts.min()) == 0:
            raise ValueError("each batch item must route a positive number of query tokens")

        memory = self.memory_encoder(memory_latents)
        expected_validity = (batch, 1, 1, self.config.memory_tokens)
        if memory_validity_mask is None:
            memory_validity_mask = torch.ones(expected_validity, dtype=torch.bool, device=memory.device)
        if memory_validity_mask.dtype != torch.bool or tuple(memory_validity_mask.shape) != expected_validity:
            raise ValueError(f"memory_validity_mask must be bool {expected_validity}")
        if not memory_validity_mask.flatten(1).any(dim=1).all():
            raise ValueError("enabled samples must have at least one valid memory token")

        k = self.norm_k_mem(self.wk_mem(memory)).view(
            batch, self.config.memory_tokens, self.config.num_heads, self.config.head_dim
        )
        v = self.wv_mem(memory).view(
            batch, self.config.memory_tokens, self.config.num_heads, self.config.head_dim
        )
        scattered = torch.zeros_like(h_base)
        fused_delta = torch.zeros_like(h_base)
        max_queries = int(counts.max())
        padded_weights = torch.zeros(
            batch, self.config.num_heads, max_queries, self.config.memory_tokens,
            dtype=torch.float32, device=h_base.device,
        )
        query_validity = torch.zeros(batch, max_queries, dtype=torch.bool, device=h_base.device)
        routed_memory_deltas = []
        routed_bases = []
        routed_fused_values = []
        bridge_records: list[dict[str, Tensor]] = []
        norm_q_weight = getattr(self._base_norm_q, "weight", None)
        norm_q_eps = float(getattr(self._base_norm_q, "eps", self.config.eps))
        for batch_index in range(batch):
            indices = route_query_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
            routed_query = h_sa0[batch_index : batch_index + 1, indices]
            q_projection = F.linear(
                routed_query,
                self._base_q.weight.detach(),
                None if self._base_q.bias is None else self._base_q.bias.detach(),
            )
            q_normed = q_projection.float() * torch.rsqrt(
                q_projection.float().pow(2).mean(dim=-1, keepdim=True) + norm_q_eps
            )
            if norm_q_weight is not None:
                q_normed = q_normed.to(q_projection.dtype) * norm_q_weight.detach()
            else:
                q_normed = q_normed.to(q_projection.dtype)
            q = q_normed.view(1, -1, self.config.num_heads, self.config.head_dim)
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", q.float(), k[batch_index : batch_index + 1].float()
            ) / math.sqrt(self.config.head_dim)
            logits = logits.masked_fill(
                ~memory_validity_mask[batch_index : batch_index + 1], float("-inf")
            )
            weights = torch.softmax(logits, dim=-1)
            attended = torch.einsum(
                "bhqk,bkhd->bqhd", weights, v[batch_index : batch_index + 1].float()
            )
            routed_delta = self.wo_mem(attended.to(v.dtype).flatten(2))
            routed_base = h_base[batch_index : batch_index + 1, indices]
            routed_fused, bridge_diag = self.bridge(routed_delta, routed_base)
            scattered[batch_index, indices] = routed_delta[0].to(scattered.dtype)
            fused_delta[batch_index, indices] = (routed_fused - routed_base)[0]
            count = indices.numel()
            padded_weights[batch_index, :, :count] = weights[0]
            query_validity[batch_index, :count] = True
            routed_memory_deltas.append(routed_delta)
            routed_bases.append(routed_base)
            routed_fused_values.append(routed_fused)
            bridge_records.append(bridge_diag)
        if torch.count_nonzero(scattered[~route_query_mask]).item() != 0:
            raise RuntimeError("non-support memory delta is not exact zero")
        fused = h_base + fused_delta
        if torch.count_nonzero(fused_delta[~route_query_mask]).item() != 0:
            raise RuntimeError("bridge changed non-support tokens")
        if return_diagnostics:
            return fused, {
                "physical_bypass": False,
                "memory_tokens": memory,
                "attention_weights": padded_weights,
                "attention_query_validity": query_validity,
                "scattered_memory_delta": scattered,
                "fused_delta": fused_delta,
                "routed_memory_delta": routed_memory_deltas,
                "routed_base": routed_bases,
                "routed_fused": routed_fused_values,
                "bridge_records": bridge_records,
            }
        return fused

    def trainable_inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name, "shape": list(parameter.shape),
                "dtype": str(parameter.dtype), "numel": parameter.numel(),
            }
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return self.state_dict()

    def load_adapter_state_dict(self, state: Mapping[str, Tensor], strict: bool = True) -> None:
        self.load_state_dict(state, strict=strict)


def tensor_module_fingerprint(module: nn.Module) -> str:
    """Deterministic full state hash for A0-sized modules (not 28GB Wan)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class WanCausalMemoryAdapterHooks(AbstractContextManager):
    """Temporary real-Wan block-0/pre-head integration point.

    It captures the exact modulated block-0 self-attention input from the real
    block call and replaces only the input to ``model.head``. The adapter is
    therefore never written back into blocks 1..N. Hooks are removed on exit.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: CausalMemoryAdapter,
        *,
        memory_latents: Optional[Tensor],
        route_query_mask: Optional[Tensor],
        memory_validity_mask: Optional[Tensor] = None,
        adapter_enabled: bool = True,
        rejected: bool = False,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.kwargs = {
            "memory_latents": memory_latents,
            "route_query_mask": route_query_mask,
            "memory_validity_mask": memory_validity_mask,
            "adapter_enabled": adapter_enabled,
            "rejected": rejected,
            "return_diagnostics": True,
        }
        self.handles = []
        self.h_sa0: Optional[Tensor] = None
        self.pre_head_input: Optional[Tensor] = None
        self.pre_head_fused: Optional[Tensor] = None
        self.block0_output: Optional[Tensor] = None
        self.adapter_diagnostics: Optional[Mapping[str, Tensor | bool]] = None

    def __enter__(self) -> "WanCausalMemoryAdapterHooks":
        block0 = self.model.blocks[0]

        def block_pre_hook(_module, args):
            x, e = args[0], args[1]
            with torch.amp.autocast("cuda", dtype=torch.float32):
                modulation = (_module.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            self.h_sa0 = (
                _module.norm1(x).float() * (1 + modulation[1].squeeze(2))
                + modulation[0].squeeze(2)
            )

        def block_hook(_module, _args, output):
            self.block0_output = output

        def head_pre_hook(_module, args):
            if self.h_sa0 is None:
                raise RuntimeError("block-0 modulated query source was not captured")
            self.pre_head_input = args[0]
            fused, diagnostics = self.adapter(
                self.h_sa0, args[0], **self.kwargs
            )
            self.pre_head_fused = fused
            self.adapter_diagnostics = diagnostics
            return (fused, *args[1:])

        self.handles = [
            block0.register_forward_pre_hook(block_pre_hook),
            block0.register_forward_hook(block_hook),
            self.model.head.register_forward_pre_hook(head_pre_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return False


def expected_trainable_inventory(
    config: CausalMemoryAdapterConfig,
    *,
    dtype: str,
) -> list[dict[str, Any]]:
    """Unique frozen A0/A1 trainable tensor inventory, in state order."""
    d, c, e = config.hidden_dim, config.latent_channels, config.encoder_channels
    bottleneck = d // 4
    specs = [
        ("memory_encoder.conv.weight", [e, c, 3, 3]),
        ("memory_encoder.conv.bias", [e]),
        ("memory_encoder.projection.weight", [d, e]),
        ("memory_encoder.projection.bias", [d]),
        ("memory_encoder.norm.weight", [d]),
        ("wk_mem.weight", [d, d]), ("wk_mem.bias", [d]),
        ("wv_mem.weight", [d, d]), ("wv_mem.bias", [d]),
        ("wo_mem.weight", [d, d]), ("wo_mem.bias", [d]),
        ("norm_k_mem.weight", [d]),
        ("bridge.norm.weight", [d]),
        ("bridge.w1.weight", [bottleneck, d]),
        ("bridge.w1.bias", [bottleneck]),
        ("bridge.w2.weight", [d, bottleneck]),
        ("bridge.w2.bias", [d]),
    ]
    return [
        {
            "name": name, "shape": shape, "dtype": dtype,
            "numel": math.prod(shape),
        }
        for name, shape in specs
    ]
